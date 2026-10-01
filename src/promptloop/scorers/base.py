"""Scorer protocol and shared text normalisation."""

from typing import Protocol

# Stripped from both ends before comparing: whitespace, quotes, backticks, markdown
# emphasis and trailing sentence punctuation. Not '-' or '+', which can carry meaning.
_EDGE_CHARS = " \t\r\n\"'`.,!?;:*“”‘’"


class Scorer(Protocol):
    name: str

    def score(self, output: str, expected: str) -> float:
        """Score a model output against the expected output, in [0, 1]."""
        ...


def normalise(text: str) -> str:
    """Lowercase, collapse internal whitespace, strip edge quotes/punctuation."""
    return " ".join(text.lower().split()).strip(_EDGE_CHARS)
