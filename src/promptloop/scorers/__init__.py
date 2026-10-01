"""Scorers: map (output, expected) to a score in [0, 1]."""

from promptloop.models import ScorerName
from promptloop.scorers.base import Scorer, normalise
from promptloop.scorers.exact import ExactScorer
from promptloop.scorers.json_match import JsonMatchScorer

__all__ = ["ExactScorer", "JsonMatchScorer", "Scorer", "get_scorer", "normalise"]


def get_scorer(name: ScorerName) -> Scorer:
    """Scorer for a task's `scorer:` field."""
    match name:
        case "exact":
            return ExactScorer()
        case "json_match":
            return JsonMatchScorer()
        case "judge":
            raise NotImplementedError("the 'judge' scorer arrives in M6")
