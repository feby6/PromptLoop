"""Exact match, optionally after normalisation."""

from promptloop.scorers.base import normalise


class ExactScorer:
    name = "exact"

    def __init__(self, normalised: bool = True) -> None:
        self.normalised = normalised

    def score(self, output: str, expected: str) -> float:
        if self.normalised:
            output, expected = normalise(output), normalise(expected)
        return 1.0 if output == expected else 0.0
