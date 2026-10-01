import json

import pytest

from promptloop.scorers import ExactScorer, JsonMatchScorer, get_scorer, normalise
from promptloop.scorers.json_match import _MISSING, extract_json


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Positive", "positive"),
        ("  positive.\n", "positive"),
        ('"positive"', "positive"),
        ("**Positive**", "positive"),
        ("`neutral`!", "neutral"),
        ("New   York\nCity", "new york city"),
        ("-5", "-5"),
        ("", ""),
    ],
)
def test_normalise(raw: str, expected: str) -> None:
    assert normalise(raw) == expected


def test_exact_normalised() -> None:
    s = ExactScorer()
    assert s.score("Positive.", "positive") == 1.0
    assert s.score("negative", "positive") == 0.0
    assert s.score("positive sentiment", "positive") == 0.0


def test_exact_strict() -> None:
    s = ExactScorer(normalised=False)
    assert s.score("positive", "positive") == 1.0
    assert s.score("Positive", "positive") == 0.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Sure! Here you go:\n```\n{"a": 1}\n```\nAnything else?', {"a": 1}),
        ('The answer is {"a": {"b": 2}} as requested.', {"a": {"b": 2}}),
        ("[1, 2]", [1, 2]),
        ("null", None),
    ],
)
def test_extract_json(text: str, expected: object) -> None:
    assert extract_json(text) == expected


def test_extract_json_failure() -> None:
    assert extract_json("no json here {oops") is _MISSING


def _j(obj: object) -> str:
    return json.dumps(obj)


def test_json_per_field_accuracy() -> None:
    s = JsonMatchScorer()
    expected = _j({"name": "Ada Lovelace", "year": 1815, "field": "maths"})
    assert s.score('{"name": "ada lovelace", "year": 1815.0, "field": "maths"}', expected) == 1.0
    assert s.score('{"name": "Ada Lovelace", "year": 1816}', expected) == pytest.approx(1 / 3)
    assert (
        s.score('{"name": "Ada Lovelace", "year": 1815, "field": "maths", "x": 1}', expected) == 1.0
    )


def test_json_invalid_or_wrong_shape_scores_zero() -> None:
    s = JsonMatchScorer()
    assert s.score("I can't answer that.", _j({"a": 1})) == 0.0
    assert s.score("[1]", _j({"a": 1})) == 0.0


def test_json_type_strictness() -> None:
    s = JsonMatchScorer()
    assert s.score('{"a": "1"}', _j({"a": 1})) == 0.0
    assert s.score('{"a": 1}', _j({"a": True})) == 0.0
    assert s.score('{"a": null}', _j({"a": None})) == 1.0


def test_json_nested_and_lists_must_match_entirely() -> None:
    s = JsonMatchScorer()
    expected = _j({"tags": ["A", "b"], "meta": {"k": "v"}})
    assert s.score(_j({"tags": ["a", "B"], "meta": {"k": "V"}}), expected) == 1.0
    assert s.score(_j({"tags": ["a"], "meta": {"k": "v", "extra": 1}}), expected) == 0.0


def test_json_non_object_expected() -> None:
    s = JsonMatchScorer()
    assert s.score("```json\n[1, 2, 3]\n```", _j([1, 2, 3])) == 1.0
    assert s.score("[1, 2]", _j([1, 2, 3])) == 0.0
    assert s.score("{}", _j({})) == 1.0


def test_get_scorer() -> None:
    assert get_scorer("exact").name == "exact"
    assert get_scorer("json_match").name == "json_match"
    with pytest.raises(NotImplementedError):
        get_scorer("judge")
