"""LLM-as-judge: grade an output against the reference with a rubric.

Noisier and costlier than deterministic scorers (one extra LLM call per example), so it
is only chosen for free-form outputs where exact/JSON matching can't work.
"""

import re

from promptloop.llm import LLMClient
from promptloop.models import LLMRequest, Message
from promptloop.prompts import render
from promptloop.scorers.json_match import extract_json

DEFAULT_RUBRIC = (
    "The output is correct and complete for the task, matches the reference in meaning, "
    "and follows the format the task requires."
)
_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*(\d+(?:\.\d+)?)', re.IGNORECASE)


def parse_judge_score(text: str) -> float | None:
    """0-10 judge score from the reply, normalised to [0, 1]; None if unreadable."""
    data = extract_json(text)
    raw: object = data.get("score") if isinstance(data, dict) else None
    if raw is None:
        # Fall back to a `score: 7` pattern when the judge didn't return clean JSON.
        match = _SCORE_RE.search(text)
        raw = match.group(1) if match else None
    if isinstance(raw, str):
        try:
            raw = float(raw)
        except ValueError:
            return None
    if not isinstance(raw, int | float) or isinstance(raw, bool):
        return None
    return min(max(float(raw), 0.0), 10.0) / 10.0


class JudgeError(RuntimeError):
    """The judge's reply had no readable score."""


class JudgeScorer:
    name = "judge"

    def __init__(self, llm: LLMClient, model: str, task_description: str, rubric: str | None):
        self.llm = llm
        self.model = model
        self.task_description = task_description
        self.rubric = rubric or DEFAULT_RUBRIC

    async def ascore(self, output: str, expected: str, input: str) -> float:
        prompt = render(
            "judge",
            task=self.task_description,
            rubric=self.rubric,
            input=input,
            expected=expected,
            output=output,
        )
        # Temperature 0 keeps grades as repeatable as the provider allows, and lets the
        # cache make re-grading an identical output free.
        response = await self.llm.acomplete(
            LLMRequest(model=self.model, messages=[Message(role="user", content=prompt)])
        )
        score = parse_judge_score(response.content)
        if score is None:
            raise JudgeError(f"judge returned no score: {response.content[:200]!r}")
        return score
