"""Test doubles for the LLM layer."""

import json
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


def loop_responder(
    target: Callable[[str, str], str],
    initial: list[str] | None = None,
    refined: list[str] | None = None,
) -> Callable[[dict[str, Any]], Any]:
    """A fake model that plays every role in the optimisation loop.

    Target-model calls are recognised by their system message (the candidate prompt) and
    answered by `target(system_prompt, user_input)`. Optimiser calls are single user
    messages, recognised by phrases from the templates in src/promptloop/prompts/.
    """
    initial = initial or ["Answer the question.", "Be brief.", "Reply with one word."]
    refined = refined or ["v2: always reply yes.", "v2: reply yes, nothing else."]

    def respond(kwargs: dict[str, Any]) -> Any:
        messages = kwargs["messages"]
        if messages[0]["role"] == "system":
            return target(messages[0]["content"], messages[1]["content"])
        text = messages[0]["content"]
        if "different system prompts" in text:
            return json.dumps(initial)
        if "improved system prompts" in text:
            return "Here you go:\n```json\n" + json.dumps(refined) + "\n```"
        if "Diagnose why" in text:
            return "The prompt never says to answer yes."
        if "evaluation data" in text:
            return json.dumps([{"input": "new q", "expected_output": "yes"}])
        if "grading" in text:
            return '{"reason": "fine", "score": 8}'
        raise AssertionError(f"unexpected optimiser prompt: {text[:80]}")

    return respond
