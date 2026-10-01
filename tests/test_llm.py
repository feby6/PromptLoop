import asyncio
from typing import Any

import httpx
import litellm
import pytest
from diskcache import Cache
from fakes import FakeClock, FakeCompletion

from promptloop.config import ProviderLimits, Settings
from promptloop.llm import (
    LLMClient,
    LLMError,
    RateLimiter,
    estimate_tokens,
    strip_reasoning,
)
from promptloop.models import LLMRequest, Message


def rate_limit_error(retry_after: str | None = None) -> litellm.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(
        429, headers=headers, request=httpx.Request("POST", "https://example.test")
    )
    return litellm.RateLimitError("slow down", "groq", "groq/target", response=response)


def request(content: str = "hi", **kw: Any) -> LLMRequest:
    return LLMRequest(model="groq/target", messages=[Message(role="user", content=content)], **kw)


def client(settings: Settings, cache: Cache, fake: FakeCompletion, clock: FakeClock) -> LLMClient:
    return LLMClient(
        settings, cache=cache, completion_fn=fake, clock=clock, sleep=clock.sleep, rng=lambda: 1.0
    )


def test_returns_content_and_usage(settings: Settings, cache: Cache) -> None:
    fake, clock = FakeCompletion("positive"), FakeClock()
    resp = client(settings, cache, fake, clock).complete(request())
    assert resp.content == "positive"
    assert (resp.model, resp.requested_model) == ("groq/target", "groq/target")
    assert (resp.prompt_tokens, resp.completion_tokens, resp.cached) == (10, 2, False)
    assert fake.calls[0]["num_retries"] == 0


def test_second_identical_call_is_cached(settings: Settings, cache: Cache) -> None:
    fake, clock = FakeCompletion("a", "b"), FakeClock()
    llm = client(settings, cache, fake, clock)
    first, second = llm.complete(request()), llm.complete(request())
    assert len(fake.calls) == 1
    assert second.content == first.content == "a" and second.cached


def test_sample_id_bypasses_cache(settings: Settings, cache: Cache) -> None:
    fake, clock = FakeCompletion("a", "b"), FakeClock()
    llm = client(settings, cache, fake, clock)
    assert llm.complete(request(sample_id=0)).content == "a"
    assert llm.complete(request(sample_id=1)).content == "b"


def test_retries_rate_limit_with_exponential_backoff(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion(rate_limit_error(), rate_limit_error(), "ok")
    clock = FakeClock()
    resp = client(settings, cache, fake, clock).complete(request())
    assert resp.content == "ok" and len(fake.calls) == 3
    assert clock.sleeps == [1.0, 2.0]  # base 1s, doubled; rng=1.0 -> full delay


def test_honours_retry_after_header(settings: Settings, cache: Cache) -> None:
    fake, clock = FakeCompletion(rate_limit_error(retry_after="7"), "ok"), FakeClock()
    client(settings, cache, fake, clock).complete(request())
    assert clock.sleeps == [7.0]


def test_non_retryable_error_raises_immediately(settings: Settings, cache: Cache) -> None:
    auth = litellm.AuthenticationError("bad key", "groq", "groq/target")
    fake, clock = FakeCompletion(auth), FakeClock()
    with pytest.raises(LLMError, match="bad key"):
        client(settings, cache, fake, clock).complete(request())
    assert len(fake.calls) == 1 and clock.sleeps == []


def test_exhausted_retries_fall_back(settings: Settings, cache: Cache) -> None:
    settings.fallbacks["groq/target"] = ["gemini/backup"]
    fake = FakeCompletion(*[rate_limit_error()] * 4, "from backup")
    resp = client(settings, cache, fake, FakeClock()).complete(request())
    assert resp.content == "from backup"
    assert resp.model == "gemini/backup" and resp.requested_model == "groq/target"
    assert fake.models == ["groq/target"] * 4 + ["gemini/backup"]


def test_long_retry_after_falls_back_without_waiting(settings: Settings, cache: Cache) -> None:
    settings.fallbacks["groq/target"] = ["gemini/backup"]
    fake = FakeCompletion(rate_limit_error(retry_after="3600"), "from backup")
    clock = FakeClock()
    resp = client(settings, cache, fake, clock).complete(request())
    assert resp.model == "gemini/backup" and clock.sleeps == []


def test_daily_cap_falls_back(settings: Settings, cache: Cache) -> None:
    settings.providers["groq"] = ProviderLimits(rpd=1)
    settings.fallbacks["groq/target"] = ["gemini/backup"]
    fake = FakeCompletion("first", "second")
    llm = client(settings, cache, fake, FakeClock())
    assert llm.complete(request("one")).model == "groq/target"
    assert llm.complete(request("two")).model == "gemini/backup"
    assert fake.models == ["groq/target", "gemini/backup"]


def test_daily_cap_without_fallback_raises(settings: Settings, cache: Cache) -> None:
    settings.providers["groq"] = ProviderLimits(rpd=1)
    fake = FakeCompletion()
    llm = client(settings, cache, fake, FakeClock())
    llm.complete(request("one"))
    with pytest.raises(LLMError, match="no model available"):
        llm.complete(request("two"))
    assert len(fake.calls) == 1


def test_rpm_limit_waits_for_window() -> None:
    clock = FakeClock()
    limiter = RateLimiter(ProviderLimits(rpm=2, concurrency=5), clock, clock.sleep)

    async def run() -> None:
        for _ in range(3):
            async with limiter.slot(1):
                pass

    asyncio.run(run())
    assert clock.sleeps == [60.0]


def test_tpm_limit_uses_actual_usage() -> None:
    clock = FakeClock()
    limiter = RateLimiter(ProviderLimits(tpm=100, concurrency=5), clock, clock.sleep)

    async def run() -> None:
        async with limiter.slot(10) as usage:
            usage.tokens = 90  # call used more than estimated
        clock.now = 30.0
        async with limiter.slot(20):  # 90 + 20 > 100 -> wait until t=60
            pass

    asyncio.run(run())
    assert clock.sleeps == [30.0]


def test_concurrency_limit() -> None:
    clock = FakeClock()
    limiter = RateLimiter(ProviderLimits(concurrency=2), clock, clock.sleep)
    active = peak = 0

    async def worker() -> None:
        nonlocal active, peak
        async with limiter.slot(1):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    async def run() -> None:
        await asyncio.gather(*(worker() for _ in range(6)))

    asyncio.run(run())
    assert peak == 2


def test_strip_reasoning() -> None:
    assert strip_reasoning("<think>\nhmm, positive?\n</think>\n\npositive") == "positive"
    assert strip_reasoning("  neutral \n") == "neutral"


def test_estimate_tokens() -> None:
    assert estimate_tokens(request("x" * 400, max_tokens=50)) == 150
