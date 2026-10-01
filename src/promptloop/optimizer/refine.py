"""Refinement: new candidates from the top-k parents plus the critique (beam search)."""

from promptloop.llm import LLMClient
from promptloop.models import Candidate, Example, LLMRequest, Message, ScoredCandidate
from promptloop.optimizer.common import clip, format_examples, parse_string_list
from promptloop.optimizer.generate import GENERATION_TEMPERATURE, candidate_id
from promptloop.prompts import render

# Parent prompts are shown in full up to this length; they are the thing being edited.
MAX_PARENT_CHARS = 3000


def format_parents(parents: list[ScoredCandidate]) -> str:
    return "\n\n".join(
        f"Prompt {i + 1} (score {p.score:.2f}):\n"
        f"<prompt>\n{clip(p.candidate.prompt, MAX_PARENT_CHARS)}\n</prompt>"
        for i, p in enumerate(parents)
    )


async def refine_candidates(
    llm: LLMClient,
    model: str,
    task_description: str,
    parents: list[ScoredCandidate],
    critique: str,
    train: list[Example],
    n: int,
    iteration: int,
) -> list[Candidate]:
    """`n` new candidates for `iteration`, each recording its parents for the lineage view.

    `parents[0]` is the best (and critiqued) candidate; it is listed first in every
    child's `parent_ids`, which is what the report follows to draw the lineage.
    """
    prompt = render(
        "refine",
        n=n,
        task=task_description,
        parents=format_parents(parents),
        critique=critique or "No failures on the training examples; improve robustness.",
        examples=format_examples(train),
    )
    response = await llm.acomplete(
        LLMRequest(
            model=model,
            messages=[Message(role="user", content=prompt)],
            temperature=GENERATION_TEMPERATURE,
        )
    )
    parent_ids = [p.candidate.id for p in parents]
    return [
        Candidate(
            id=candidate_id(iteration, i), prompt=p, iteration=iteration, parent_ids=parent_ids
        )
        for i, p in enumerate(parse_string_list(response.content)[:n])
    ]
