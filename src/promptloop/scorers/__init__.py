"""Scorers: map (output, expected) to a score in [0, 1]."""

import json

from promptloop.llm import LLMClient
from promptloop.models import Example, ScorerName, Task
from promptloop.scorers.base import AsyncScorer, Scorer, normalise
from promptloop.scorers.exact import ExactScorer
from promptloop.scorers.json_match import JsonMatchScorer
from promptloop.scorers.judge import JudgeScorer

__all__ = [
    "AsyncScorer",
    "ExactScorer",
    "JsonMatchScorer",
    "JudgeScorer",
    "Scorer",
    "auto_scorer_name",
    "get_scorer",
    "normalise",
    "scorer_for_task",
]

# Outputs up to this many words are treated as labels/short answers for exact matching.
SHORT_ANSWER_WORDS = 5


def get_scorer(name: ScorerName) -> Scorer:
    """Deterministic scorer by name. The judge needs an LLM: use `scorer_for_task`."""
    match name:
        case "exact":
            return ExactScorer()
        case "json_match":
            return JsonMatchScorer()
        case "judge":
            raise ValueError("the judge scorer needs an LLM; use scorer_for_task()")


def scorer_for_task(task: Task, llm: LLMClient) -> Scorer | AsyncScorer:
    """Scorer for a task. The judge runs on the optimiser model (in the web app: the user's)."""
    if task.scorer == "judge":
        return JudgeScorer(llm, task.optimizer_model, task.description, task.rubric)
    return get_scorer(task.scorer)


def auto_scorer_name(examples: list[Example]) -> ScorerName:
    """Pick a scorer from what the expected outputs look like (used by the web form).

    All JSON objects/arrays -> json_match; all short answers -> exact; otherwise the
    outputs are free-form text and only a judge can grade them.
    """
    expected = [e.expected_output for e in examples]
    if expected and all(_is_json_container(x) for x in expected):
        return "json_match"
    if all(len(x.split()) <= SHORT_ANSWER_WORDS for x in expected):
        return "exact"
    return "judge"


def _is_json_container(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict | list)
    except json.JSONDecodeError:
        return False
