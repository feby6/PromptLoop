"""Test doubles for the LLM layer."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any


class FakeClock:
    """Monotonic clock that only advances when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeCompletion:
    """Stands in for litellm.acompletion and records every call.

    Outcomes are popped in order (a string is returned as content, an exception is
    raised). Once they run out, `responder(kwargs)` decides, defaulting to "ok".
    """

    def __init__(self, *outcomes: Any, responder: Callable[[dict[str, Any]], Any] | None = None):
        self.outcomes = list(outcomes)
        self.responder = responder or (lambda _: "ok")
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else self.responder(kwargs)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2),
        )

    @property
    def models(self) -> list[str]:
        return [c["model"] for c in self.calls]
