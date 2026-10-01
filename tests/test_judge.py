import asyncio

import pytest
from diskcache import Cache
from fakes import FakeClock, FakeCompletion

from promptloop.config import Settings
from promptloop.llm import LLMClient
from promptloop.models import Example, ExampleOutput, Task
from promptloop.optimizer.evaluate import evaluate
from promptloop.scorers import JudgeScorer, auto_scorer_name, scorer_for_task
from promptloop.scorers.judge import DEFAULT_RUBRIC, JudgeError, parse_judge_score


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"reason": "ok", "score": 7}', 0.7),
        ('```json\n{"score": 10, "reason": "perfect"}\n```', 1.0),
        ('{"score": "4"}', 0.4),
        ("Reasoning... final score: 6", 0.6),
        ('{"score": 15}', 1.0),  # clamped
        ('{"score": -2}', 0.0),
        ("no idea", None),
        ('{"score": true}', None),
    ],
)
def test_parse_judge_score(reply: str, expected: float | None) -> None:
    assert parse_judge_score(reply) == expected


def make_llm(settings: Settings, cache: Cache, fake: FakeCompletion) -> LLMClient:
    clock = FakeClock()
    return LLMClient(settings, cache=cache, completion_fn=fake, clock=clock, sleep=clock.sleep)


def test_judge_scores_and_shows_everything_to_the_judge(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion('{"reason": "close", "score": 8}')
    judge = JudgeScorer(make_llm(settings, cache, fake), "groq/j", "Summarise.", None)
    score = asyncio.run(judge.ascore("short summary", "reference summary", "long text"))
    assert score == 0.8
    prompt = fake.calls[0]["messages"][0]["content"]
    for part in ["Summarise.", DEFAULT_RUBRIC, "long text", "reference summary", "short summary"]:
        assert part in prompt
    assert fake.calls[0]["temperature"] == 0.0


def test_judge_unreadable_reply_raises(settings: Settings, cache: Cache) -> None:
    judge = JudgeScorer(make_llm(settings, cache, FakeCompletion("hmm")), "groq/j", "t", "r")
    with pytest.raises(JudgeError):
        asyncio.run(judge.ascore("o", "e", "i"))


def test_evaluate_with_judge_records_scoring_failures(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion('{"score": 9}', "unparseable")
    judge = JudgeScorer(make_llm(settings, cache, fake), "groq/j", "t", "r")
    outputs = [
        ExampleOutput(input="a", expected_output="x", output="o1"),
        ExampleOutput(input="b", expected_output="y", output="o2"),
    ]
    ev = asyncio.run(evaluate("c", outputs, judge))
    scores = sorted(r.score for r in ev.results)
    assert scores == [0.0, 0.9]
    assert len(ev.errors()) == 1 and "scoring failed" in (ev.errors()[0].error or "")


def test_scorer_for_task(settings: Settings, cache: Cache) -> None:
    llm = make_llm(settings, cache, FakeCompletion())
    base = {"name": "t", "description": "d", "target_model": "groq/a", "optimizer_model": "groq/b"}
    judge = scorer_for_task(Task(scorer="judge", rubric="r", **base), llm)
    assert isinstance(judge, JudgeScorer) and judge.model == "groq/b"
    assert scorer_for_task(Task(scorer="exact", **base), llm).name == "exact"


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        (['{"a": 1}', "[1, 2]"], "json_match"),
        (["positive", "Negative", "sci/tech"], "exact"),
        (["42", "a short answer here"], "exact"),
        (["positive", "This is a long free-form summary of the article."], "judge"),
        (['{"a": 1}', "plain"], "exact"),
    ],
)
def test_auto_scorer_name(outputs: list[str], expected: str) -> None:
    examples = [Example(input=str(i), expected_output=o) for i, o in enumerate(outputs)]
    assert auto_scorer_name(examples) == expected
