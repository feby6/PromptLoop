"""Critic: explain why the best candidate fails (a "textual gradient" for refinement)."""

from promptloop.llm import LLMClient
from promptloop.models import LLMRequest, Message, ScoredCandidate
from promptloop.optimizer.common import format_failures
from promptloop.prompts import render

# Low temperature: a diagnosis should be focused, not creative.
CRITIQUE_TEMPERATURE = 0.2


async def critique_candidate(
    llm: LLMClient, model: str, task_description: str, scored: ScoredCandidate
) -> str:
    """Diagnosis of the candidate's train failures, or "" if it has none."""
    failures = scored.evaluation.failures()
    if not failures:
        return ""
    prompt = render(
        "critique",
        task=task_description,
        prompt=scored.candidate.prompt,
        score=f"{scored.score:.2f}",
        failures=format_failures(failures),
    )
    response = await llm.acomplete(
        LLMRequest(
            model=model,
            messages=[Message(role="user", content=prompt)],
            temperature=CRITIQUE_TEMPERATURE,
        )
    )
    return response.content.strip()
