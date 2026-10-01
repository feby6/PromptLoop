"""JSON validity + per-field accuracy.

Expected output must be valid JSON. If it is an object, the score is the fraction of its
top-level fields the output gets right (extra output fields are ignored). Otherwise the
whole value must match. Unparseable output scores 0. Strings compare normalised,
numbers compare with a small tolerance, nested values must match entirely.
"""

import json
import math
import re
from typing import Any

from promptloop.scorers.base import normalise

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_MISSING = object()


def extract_json(text: str) -> Any:
    """Parse JSON from model output: raw, inside a ``` fence, or the outermost {...}/[...]
    span. Returns `_MISSING` when nothing parses (JSON `null` is a valid result)."""
    candidates = [text.strip(), *(m.strip() for m in _FENCE_RE.findall(text))]
    for open_ch, close_ch in ("{}", "[]"):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return _MISSING


def values_equal(got: Any, expected: Any) -> bool:
    if isinstance(expected, str) and isinstance(got, str):
        return normalise(got) == normalise(expected)
    if _is_number(expected) and _is_number(got):
        return math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(expected, dict) and isinstance(got, dict):
        return expected.keys() == got.keys() and all(
            values_equal(got[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list) and isinstance(got, list):
        return len(expected) == len(got) and all(map(values_equal, got, expected))
    return got == expected and type(got) is type(expected)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


class JsonMatchScorer:
    name = "json_match"

    def score(self, output: str, expected: str) -> float:
        want = json.loads(expected)  # validated at load time; a failure here is a data bug
        got = extract_json(output)
        if got is _MISSING:
            return 0.0
        if not isinstance(want, dict) or not want:
            return 1.0 if values_equal(got, want) else 0.0
        if not isinstance(got, dict):
            return 0.0
        hits = sum(k in got and values_equal(got[k], v) for k, v in want.items())
        return hits / len(want)
