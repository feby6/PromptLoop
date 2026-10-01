"""LiteLLM wrapper. Every LLM call in PromptLoop goes through `LLMClient`.

Per call: cache lookup -> for each model in [requested, *fallbacks]: daily-cap check,
RPM/TPM/concurrency limits, call with exponential backoff on retryable errors.
A model that hits its daily cap or keeps returning 429 hands over to the next fallback.
"""

import asyncio
import logging
import random
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import litellm
from diskcache import Cache

from promptloop.config import ProviderLimits, Settings, provider_of
from promptloop.models import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)
litellm.suppress_debug_info = True

CompletionFn = Callable[..., Awaitable[Any]]
Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]

WINDOW_S = 60.0
DAILY_KEY_TTL_S = 2 * 24 * 3600
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
    litellm.Timeout,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    """An LLM call failed and could not be recovered."""


class DailyCapReached(LLMError):
    """The provider's configured requests-per-day limit is used up."""


class ProviderRateLimited(LLMError):
    """The provider kept returning 429 after all retries."""


def strip_reasoning(text: str) -> str:
    """Remove `<think>...</think>` blocks some reasoning models put in their output."""
    return _THINK_RE.sub("", text).strip()


def estimate_tokens(request: LLMRequest) -> int:
    """Rough pre-call token estimate (~4 chars/token) used for TPM budgeting."""
    prompt = sum(len(m.content) for m in request.messages) // 4
    return prompt + (request.max_tokens or 256)


def _retry_after_s(error: Exception) -> float | None:
    headers = getattr(error, "headers", None) or getattr(
        getattr(error, "response", None), "headers", None
    )
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


@dataclass
class _TokenEntry:
    at: float
    tokens: int


class RateLimiter:
    """Sliding-window RPM and TPM limits plus a concurrency cap for one provider."""

    def __init__(self, limits: ProviderLimits, clock: Clock, sleep: Sleep) -> None:
        self._limits = limits
        self._clock = clock
        self._sleep = sleep
        self._requests: deque[float] = deque()
        self._tokens: deque[_TokenEntry] = deque()
        self._loop: asyncio.AbstractEventLoop | None = None

    def _bind_loop(self) -> None:
        # asyncio primitives belong to one event loop; `complete()` starts a new loop per call.
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self._lock = asyncio.Lock()
            self._semaphore = asyncio.Semaphore(self._limits.concurrency)

    def _prune(self, now: float) -> None:
        while self._requests and now - self._requests[0] >= WINDOW_S:
            self._requests.popleft()
        while self._tokens and now - self._tokens[0].at >= WINDOW_S:
            self._tokens.popleft()

    def _wait_s(self, now: float, est_tokens: int) -> float:
        wait = 0.0
        rpm, tpm = self._limits.rpm, self._limits.tpm
        if rpm and len(self._requests) >= rpm:
            wait = max(wait, self._requests[0] + WINDOW_S - now)
        # An empty window always admits, so one oversized request can't block forever.
        if tpm and self._tokens and sum(e.tokens for e in self._tokens) + est_tokens > tpm:
            wait = max(wait, self._tokens[0].at + WINDOW_S - now)
        return wait

    @asynccontextmanager
    async def slot(self, est_tokens: int) -> AsyncIterator[_TokenEntry]:
        """Wait for capacity, then hold a concurrency slot. Set `.tokens` to actual usage."""
        self._bind_loop()
        async with self._semaphore:
            async with self._lock:
                while True:
                    now = self._clock()
                    self._prune(now)
                    wait = self._wait_s(now, est_tokens)
                    if wait <= 0:
                        break
                    logger.debug("rate limit: waiting %.1fs", wait)
                    await self._sleep(wait)
                self._requests.append(now)
                entry = _TokenEntry(at=now, tokens=est_tokens)
                self._tokens.append(entry)
            yield entry


class LLMClient:
    """Cached, rate-limited, retrying LLM client. Dependencies are injectable for tests."""

    def __init__(
        self,
        settings: Settings,
        cache: Cache | None = None,
        completion_fn: CompletionFn | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._settings = settings
        self._cache = cache if cache is not None else Cache(str(settings.cache_dir / "llm"))
        self._completion = completion_fn or litellm.acompletion
        self._clock = clock
        self._sleep = sleep
        self._rng = rng
        self._limiters: dict[str, RateLimiter] = {}

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper around `acomplete`."""
        return asyncio.run(self.acomplete(request))

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        key = f"resp:{request.cache_key()}"
        hit = self._cache.get(key)
        if hit is not None:
            return LLMResponse.model_validate(hit).model_copy(update={"cached": True})

        chain = [request.model, *self._settings.fallbacks.get(request.model, [])]
        last_error: LLMError | None = None
        for model in chain:
            try:
                response = await self._call_with_retries(model, request)
            except (DailyCapReached, ProviderRateLimited) as e:
                logger.warning("%s unavailable: %s", model, e)
                last_error = e
                continue
            self._cache.set(key, response.model_dump())
            return response
        raise LLMError(f"no model available for {request.model}: {last_error}") from last_error

    def close(self) -> None:
        self._cache.close()

    def _limiter(self, provider: str) -> RateLimiter:
        if provider not in self._limiters:
            limits = self._settings.limits_for(provider)
            self._limiters[provider] = RateLimiter(limits, self._clock, self._sleep)
        return self._limiters[provider]

    def _count_daily_request(self, provider: str) -> None:
        rpd = self._settings.limits_for(provider).rpd
        if not rpd:
            return
        key = f"rpd:{provider}:{datetime.now(UTC).date().isoformat()}"
        self._cache.add(key, 0, expire=DAILY_KEY_TTL_S)
        if self._cache.incr(key) > rpd:
            raise DailyCapReached(f"{provider}: {rpd} requests/day used")

    def _backoff_s(self, attempt: int) -> float:
        retry = self._settings.retry
        base = min(retry.backoff_max_s, retry.backoff_base_s * 2**attempt)
        return base * (0.5 + self._rng() / 2)  # jitter in [0.5, 1.0) x base

    async def _call_with_retries(self, model: str, request: LLMRequest) -> LLMResponse:
        provider = provider_of(model)
        limiter = self._limiter(provider)
        retry = self._settings.retry
        error: Exception | None = None
        for attempt in range(retry.max_retries + 1):
            self._count_daily_request(provider)
            async with limiter.slot(estimate_tokens(request)) as usage:
                try:
                    raw = await self._completion(
                        model=model,
                        messages=[m.model_dump() for m in request.messages],
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        timeout=retry.request_timeout_s,
                        num_retries=0,
                    )
                except RETRYABLE_ERRORS as e:
                    error = e
                except Exception as e:
                    raise LLMError(f"{model}: {e}") from e
                else:
                    response = _to_response(raw, model, request.model)
                    usage.tokens = response.prompt_tokens + response.completion_tokens
                    return response

            retry_after = _retry_after_s(error)
            if retry_after is not None and retry_after > retry.backoff_max_s:
                break  # provider says come back much later: don't wait, fall back
            if attempt == retry.max_retries:
                break
            delay = max(self._backoff_s(attempt), retry_after or 0.0)
            logger.info(
                "%s: %s; retry %d in %.1fs", model, type(error).__name__, attempt + 1, delay
            )
            await self._sleep(delay)

        if isinstance(error, litellm.RateLimitError):
            raise ProviderRateLimited(f"{model}: rate limited after retries") from error
        raise LLMError(f"{model}: {error}") from error


def _to_response(raw: Any, model: str, requested_model: str) -> LLMResponse:
    content = raw.choices[0].message.content or ""
    usage = getattr(raw, "usage", None)
    return LLMResponse(
        content=strip_reasoning(content),
        model=model,
        requested_model=requested_model,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
