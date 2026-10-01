import asyncio
import json

import pytest
from diskcache import Cache
from fakes import FakeClock, FakeCompletion, loop_responder

from promptloop.config import Settings
from promptloop.llm import LLMClient
from promptloop.models import (
    Candidate,
    EvalResult,
    Example,
    ExampleResult,
    ScoredCandidate,
)
from promptloop.optimizer.common import OptimizerError, parse_examples, parse_string_list
from promptloop.optimizer.critique import critique_candidate
from promptloop.optimizer.generate import generate_candidates
from promptloop.optimizer.refine import refine_candidates
from promptloop.optimizer.synthesize import synthesize_examples
from promptloop.prompts import render

TRAIN = [Example(input=f"q{i}", expected_output="yes") for i in range(3)]


def make_llm(settings: Settings, cache: Cache, fake: FakeCompletion) -> LLMClient:
    clock = FakeClock()
    return LLMClient(settings, cache=cache, completion_fn=fake, clock=clock, sleep=clock.sleep)


def scored(cid: str, scores: list[float], prompt: str = "p") -> ScoredCandidate:
    results = [
        ExampleResult(input=f"q{i}", expected_output="yes", output="no", score=s)
        for i, s in enumerate(scores)
    ]
    return ScoredCandidate(
        candidate=Candidate(id=cid, prompt=prompt),
        evaluation=EvalResult(candidate_id=cid, results=results),
    )


# --- templates ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("generate", {"n": 3, "task": "t", "examples": "e"}),
        ("critique", {"task": "t", "prompt": "p", "score": "0.5", "failures": "f"}),
        ("refine", {"n": 3, "task": "t", "parents": "p", "critique": "c", "examples": "e"}),
        ("synthesize", {"n": 3, "task": "t", "examples": "e"}),
        (
            "judge",
            {"task": "t", "rubric": "r", "input": "i", "expected": "x", "output": "o"},
        ),
    ],
)
def test_every_template_renders(name: str, values: dict[str, object]) -> None:
    text = render(name, **values)
    assert "$" not in text  # no placeholder left unfilled


def test_render_missing_value_raises() -> None:
    with pytest.raises(KeyError):
        render("generate", n=3)


# --- parsing --------------------------------------------------------------------------


def test_parse_string_list_variants() -> None:
    assert parse_string_list('["a", " b "]') == ["a", "b"]
    assert parse_string_list('Sure:\n```json\n["a"]\n```') == ["a"]
    assert parse_string_list('[{"prompt": "a"}, "", 3, "b"]') == ["a", "b"]
    # Braces inside a prompt must not be mistaken for the answer.
    assert parse_string_list('Here: ["Return {\\"k\\": 1}"]') == ['Return {"k": 1}']


@pytest.mark.parametrize("bad", ["no json", '{"a": 1}', "[]", '["", "  "]'])
def test_parse_string_list_rejects(bad: str) -> None:
    with pytest.raises(OptimizerError):
        parse_string_list(bad)


def test_parse_examples_skips_bad_items_and_encodes_json() -> None:
    text = json.dumps(
        [
            {"input": "a", "expected_output": "x"},
            {"input": "b", "expected_output": {"k": 1}},
            {"input": "", "expected_output": "x"},
            {"input": "c"},
            "junk",
        ]
    )
    examples = parse_examples(text)
    assert [(e.input, e.expected_output) for e in examples] == [("a", "x"), ("b", '{"k": 1}')]


# --- steps ----------------------------------------------------------------------------


def test_generate_candidates(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion(responder=loop_responder(lambda p, i: "x"))
    llm = make_llm(settings, cache, fake)
    cands = asyncio.run(generate_candidates(llm, "groq/opt", "Say yes.", TRAIN, n=2))
    assert [c.id for c in cands] == ["it0-c0", "it0-c1"]
    assert cands[0].prompt == "Answer the question." and cands[0].iteration == 0
    sent = fake.calls[0]
    assert sent["model"] == "groq/opt" and sent["temperature"] > 0
    assert "Say yes." in sent["messages"][0]["content"]


def test_critique_skips_call_without_failures(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion(responder=loop_responder(lambda p, i: "x"))
    llm = make_llm(settings, cache, fake)
    assert asyncio.run(critique_candidate(llm, "groq/opt", "t", scored("c", [1.0, 1.0]))) == ""
    assert fake.calls == []
    text = asyncio.run(critique_candidate(llm, "groq/opt", "t", scored("c", [1.0, 0.0])))
    assert "answer yes" in text
    assert "q1" in fake.calls[0]["messages"][0]["content"]  # the failing case is shown


def test_refine_records_parents(settings: Settings, cache: Cache) -> None:
    llm = make_llm(settings, cache, FakeCompletion(responder=loop_responder(lambda p, i: "x")))
    parents = [scored("best", [1.0, 0.0]), scored("second", [0.0, 0.0])]
    cands = asyncio.run(
        refine_candidates(llm, "groq/opt", "t", parents, "fix it", TRAIN, n=2, iteration=3)
    )
    assert [c.id for c in cands] == ["it3-c0", "it3-c1"]
    assert all(c.parent_ids == ["best", "second"] and c.iteration == 3 for c in cands)


def test_synthesize_drops_duplicates_of_seeds(settings: Settings, cache: Cache) -> None:
    reply = json.dumps(
        [
            {"input": "Seed One", "expected_output": "yes"},  # repeats a seed
            {"input": "fresh", "expected_output": "no"},
            {"input": "FRESH ", "expected_output": "no"},  # repeats the previous one
            {"input": "another", "expected_output": "yes"},
        ]
    )
    llm = make_llm(settings, cache, FakeCompletion(reply))
    seeds = [Example(input="seed one", expected_output="yes")]
    out = asyncio.run(synthesize_examples(llm, "groq/m", "task", seeds, n=5))
    assert [e.input for e in out] == ["fresh", "another"]
