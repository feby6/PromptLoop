import asyncio
from typing import Any

import litellm
from diskcache import Cache
from fakes import FakeClock, FakeCompletion

from promptloop.config import Settings
from promptloop.llm import LLMClient
from promptloop.models import Candidate, Example, ExampleOutput
from promptloop.optimizer.evaluate import evaluate, score_candidate
from promptloop.optimizer.execute import execute_candidate
from promptloop.scorers import ExactScorer

EXAMPLES = [
    Example(input="great product", expected_output="positive"),
    Example(input="broke in a day", expected_output="negative"),
    Example(input="it is a box", expected_output="neutral"),
]
CANDIDATE = Candidate(id="c1", prompt="Classify the sentiment.")


def keyword_responder(kwargs: dict[str, Any]) -> str:
    """Fake target model: right on two examples, wrong on the neutral one."""
    text = kwargs["messages"][-1]["content"]
    return "Positive." if "great" in text else "negative"


def make_llm(settings: Settings, cache: Cache, fake: FakeCompletion) -> LLMClient:
    clock = FakeClock()
    return LLMClient(settings, cache=cache, completion_fn=fake, clock=clock, sleep=clock.sleep)


def test_execute_sends_prompt_as_system_and_keeps_order(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion(responder=keyword_responder)
    llm = make_llm(settings, cache, fake)
    outputs = asyncio.run(execute_candidate(llm, CANDIDATE, EXAMPLES, "groq/target"))

    assert [o.input for o in outputs] == [e.input for e in EXAMPLES]
    assert [o.output for o in outputs] == ["Positive.", "negative", "negative"]
    assert all(o.error is None and not o.cached for o in outputs)
    sent = {c["messages"][1]["content"]: c["messages"] for c in fake.calls}
    assert sent["great product"][0] == {"role": "system", "content": "Classify the sentiment."}
    assert all(c["model"] == "groq/target" and c["temperature"] == 0.0 for c in fake.calls)


def test_execute_captures_failures_per_example(settings: Settings, cache: Cache) -> None:
    def responder(kwargs: dict[str, Any]) -> Any:
        if "box" in kwargs["messages"][-1]["content"]:
            return litellm.AuthenticationError("bad key", "groq", "groq/target")
        return "positive"

    llm = make_llm(settings, cache, FakeCompletion(responder=responder))
    outputs = asyncio.run(execute_candidate(llm, CANDIDATE, EXAMPLES, "groq/target"))
    assert [o.error is not None for o in outputs] == [False, False, True]
    assert outputs[2].output == "" and "bad key" in (outputs[2].error or "")


def test_rerun_is_served_from_cache(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion(responder=keyword_responder)
    llm = make_llm(settings, cache, fake)
    asyncio.run(execute_candidate(llm, CANDIDATE, EXAMPLES, "groq/target"))
    again = asyncio.run(execute_candidate(llm, CANDIDATE, EXAMPLES, "groq/target"))
    assert len(fake.calls) == 3 and all(o.cached for o in again)


def test_evaluate_scores_and_zeroes_errors() -> None:
    outputs = [
        ExampleOutput(input="a", expected_output="positive", output="Positive"),
        ExampleOutput(input="b", expected_output="negative", output="positive"),
        ExampleOutput(input="c", expected_output="neutral", output="", error="timeout"),
    ]
    ev = asyncio.run(evaluate("c1", outputs, ExactScorer()))
    assert [r.score for r in ev.results] == [1.0, 0.0, 0.0]
    assert ev.mean_score == 1 / 3
    assert [r.input for r in ev.failures()] == ["b", "c"]
    assert [r.input for r in ev.errors()] == ["c"]


def test_score_candidate_end_to_end(settings: Settings, cache: Cache) -> None:
    llm = make_llm(settings, cache, FakeCompletion(responder=keyword_responder))
    ev = asyncio.run(score_candidate(llm, CANDIDATE, EXAMPLES, "groq/target", ExactScorer()))
    assert ev.candidate_id == "c1"
    assert ev.mean_score == 2 / 3
    assert [f.input for f in ev.failures()] == ["it is a box"]
