import pytest
from pydantic import ValidationError

from promptloop.models import EvalResult, ExampleResult, LLMRequest, Message, Task


def _task(**overrides: object) -> dict[str, object]:
    return {
        "name": "t",
        "description": "d",
        "scorer": "exact",
        "target_model": "groq/a",
        "optimizer_model": "groq/b",
        **overrides,
    }


def test_judge_scorer_requires_rubric() -> None:
    with pytest.raises(ValidationError, match="rubric"):
        Task.model_validate(_task(scorer="judge"))
    assert Task.model_validate(_task(scorer="judge", rubric="be fair")).rubric == "be fair"


def test_unknown_scorer_rejected() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(_task(scorer="fuzzy"))


def test_score_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        ExampleResult(input="i", expected_output="e", output="o", score=1.5)


def test_eval_result_mean_and_failures() -> None:
    results = [
        ExampleResult(input=str(i), expected_output="e", output="o", score=s)
        for i, s in enumerate([1.0, 0.0, 0.5])
    ]
    ev = EvalResult(candidate_id="c", results=results)
    assert ev.mean_score == pytest.approx(0.5)
    assert [r.input for r in ev.failures()] == ["1", "2"]
    assert EvalResult(candidate_id="c", results=[]).mean_score == 0.0


def test_cache_key_is_stable_and_param_sensitive() -> None:
    req = LLMRequest(model="groq/a", messages=[Message(role="user", content="hi")])
    assert req.cache_key() == req.model_copy().cache_key()
    assert req.cache_key() != req.model_copy(update={"sample_id": 1}).cache_key()
    assert req.cache_key() != req.model_copy(update={"temperature": 0.7}).cache_key()
    assert req.cache_key() != req.model_copy(update={"model": "groq/b"}).cache_key()
