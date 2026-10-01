"""Helpers shared by the optimiser steps that talk to the optimiser model."""

import asyncio
import json
from collections.abc import Awaitable

from promptloop.models import Example, ExampleResult
from promptloop.scorers.json_match import NOT_JSON, extract_json

# Bounds on what goes into optimiser prompts: enough signal to generalise from, small
# enough to stay well inside free-tier token-per-minute limits.
MAX_PROMPT_EXAMPLES = 5
MAX_PROMPT_FAILURES = 5
MAX_FIELD_CHARS = 600


class OptimizerError(RuntimeError):
    """The optimiser model's reply could not be used."""


async def gather_or_cancel[T](*aws: Awaitable[T]) -> list[T]:
    """Like asyncio.gather, but on the first exception cancel everything still running.

    Plain gather leaves the other calls going, which after a quota error would keep
    spending the user's quota on a run that is already failing.
    """
    tasks = [asyncio.ensure_future(a) for a in aws]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def clip(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_examples(examples: list[Example], limit: int = MAX_PROMPT_EXAMPLES) -> str:
    return "\n\n".join(
        f"Input:\n{clip(e.input)}\nExpected output:\n{clip(e.expected_output)}"
        for e in examples[:limit]
    )


def format_failures(failures: list[ExampleResult], limit: int = MAX_PROMPT_FAILURES) -> str:
    return "\n\n".join(
        f"Input:\n{clip(f.input)}\nExpected output:\n{clip(f.expected_output)}\n"
        f"Model output:\n{clip(f.output) or '(empty)'}"
        for f in failures[:limit]
    )


def parse_string_list(text: str) -> list[str]:
    """A JSON array of non-empty strings from a model reply (fenced or in prose).

    Objects with a "prompt" key are accepted too, since models sometimes wrap each
    prompt that way despite being asked for plain strings.
    """
    data = extract_json(text)
    if data is NOT_JSON or not isinstance(data, list):
        raise OptimizerError(f"expected a JSON array of prompts, got: {clip(text, 200)!r}")
    items = [x.get("prompt") if isinstance(x, dict) else x for x in data]
    prompts = [x.strip() for x in items if isinstance(x, str) and x.strip()]
    if not prompts:
        raise OptimizerError("the optimiser returned no usable prompts")
    return prompts


def parse_examples(text: str) -> list[Example]:
    """A JSON array of {"input", "expected_output"} objects from a model reply.

    Malformed items are skipped rather than failing the batch. A JSON expected output
    given as an object instead of a string is re-encoded, because scorers compare
    against the string form.
    """
    data = extract_json(text)
    if data is NOT_JSON or not isinstance(data, list):
        raise OptimizerError(f"expected a JSON array of examples, got: {clip(text, 200)!r}")
    examples = []
    for item in data:
        if not isinstance(item, dict):
            continue
        inp, out = item.get("input"), item.get("expected_output")
        if isinstance(out, dict | list):
            out = json.dumps(out, ensure_ascii=False)
        if isinstance(inp, str) and isinstance(out, str) and inp.strip() and out.strip():
            examples.append(Example(input=inp.strip(), expected_output=out.strip()))
    return examples
