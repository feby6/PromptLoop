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
    MemoryCache,
    QuotaExhausted,
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


def test_user_key_is_passed_per_call(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion("ok")
    llm = LLMClient(settings, cache=cache, completion_fn=fake, api_keys={"groq": "user-key"})
    llm.complete(request())
    assert fake.calls[0]["api_key"] == "user-key"


def test_env_keys_can_be_disallowed(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion("ok")
    llm = LLMClient(settings, cache=cache, completion_fn=fake, allow_env_keys=False)
    with pytest.raises(LLMError, match="no API key provided for provider 'groq'"):
        llm.complete(request())
    assert fake.calls == []


def test_key_is_redacted_from_errors(settings: Settings, cache: Cache) -> None:
    secret = "sk-very-secret-123"
    fake = FakeCompletion(litellm.AuthenticationError(f"bad key {secret}", "groq", "groq/target"))
    llm = LLMClient(settings, cache=cache, completion_fn=fake, api_keys={"groq": secret})
    with pytest.raises(LLMError) as info:
        llm.complete(request())
    assert secret not in str(info.value) and "***" in str(info.value)
    assert info.value.__cause__ is None  # nothing chained that could leak it


def test_key_is_not_part_of_cache_key(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion("first", "second")
    keys = ["user-key-AAA", "user-key-BBB"]
    for key in keys:
        resp = LLMClient(
            settings, cache=cache, completion_fn=fake, api_keys={"groq": key}
        ).complete(request())
    # Same request with a different key is a cache hit, and no key is stored anywhere.
    assert resp.cached and resp.content == "first"
    stored = " ".join(f"{k}={cache.get(k)}" for k in cache.iterkeys())
    assert "user-key" not in stored


def test_usage_counts_calls_retries_and_cache_hits(settings: Settings, cache: Cache) -> None:
    fake, clock = FakeCompletion(rate_limit_error(), "ok"), FakeClock()
    llm = client(settings, cache, fake, clock)
    llm.complete(request())
    llm.complete(request())
    assert llm.usage.model_dump() == {
        "calls": 2,
        "cache_hits": 1,
        "prompt_tokens": 10,
        "completion_tokens": 2,
    }


def test_memory_cache_behaves_like_diskcache(settings: Settings) -> None:
    mem = MemoryCache()
    assert mem.add("k", 0) and not mem.add("k", 5)
    assert mem.incr("k") == 1 and mem.incr("k", 2) == 3
    assert mem.get("missing") is None and mem.get("missing", 1) == 1
    fake = FakeCompletion("a", "b")
    llm = LLMClient(settings, cache=mem, completion_fn=fake)
    assert llm.complete(request()).content == "a"
    assert llm.complete(request()).cached
    llm.close()
    assert mem.get("k") is None


def test_limiter_learns_output_length_for_tpm_budget() -> None:
    clock = FakeClock()
    limiter = RateLimiter(ProviderLimits(tpm=1000, concurrency=5), clock, clock.sleep)
    assert limiter.expected_completion == 256

    async def run() -> None:
        async with limiter.slot(100) as slot:
            limiter.record_usage(slot, prompt_tokens=50, completion_tokens=600)

    asyncio.run(run())
    assert limiter.expected_completion == 600
    assert estimate_tokens(request("x" * 400), limiter.expected_completion) == 700


def test_429_pauses_the_whole_provider() -> None:
    clock = FakeClock()
    limiter = RateLimiter(ProviderLimits(concurrency=5), clock, clock.sleep)
    limiter.pause(20.0)

    async def run() -> None:
        async with limiter.slot(1):
            pass

    asyncio.run(run())
    assert clock.sleeps == [20.0]


def test_rate_limit_retry_pauses_provider(settings: Settings, cache: Cache) -> None:
    fake, clock = FakeCompletion(rate_limit_error(retry_after="5"), "ok"), FakeClock()
    llm = client(settings, cache, fake, clock)
    llm.complete(request())
    # The retrying request slept 5s itself; the shared cooldown had expired by then.
    assert clock.sleeps == [5.0]
    assert llm._limiter("groq")._cooldown_until == 5.0


def test_quota_exhausted_is_distinct(settings: Settings, cache: Cache) -> None:
    settings.providers["groq"] = ProviderLimits(rpd=1)
    llm = client(settings, cache, FakeCompletion(), FakeClock())
    llm.complete(request("one"))
    with pytest.raises(QuotaExhausted, match="try again later"):
        llm.complete(request("two"))
