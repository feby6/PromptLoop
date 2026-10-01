"""Initial candidate prompts from the task description and a few train examples."""

from promptloop.llm import LLMClient
from promptloop.models import Candidate, Example, LLMRequest, Message
from promptloop.optimizer.common import format_examples, parse_string_list
from promptloop.prompts import render

# Some randomness so the candidates explore different phrasings and strategies.
GENERATION_TEMPERATURE = 0.7


def candidate_id(iteration: int, index: int) -> str:
    return f"it{iteration}-c{index}"


async def generate_candidates(
    llm: LLMClient, model: str, task_description: str, train: list[Example], n: int
) -> list[Candidate]:
    """`n` diverse prompts in one optimiser call (cheaper than one call per prompt).

    Only train examples are passed in: the val split must never reach the optimiser.
    """
    prompt = render("generate", n=n, task=task_description, examples=format_examples(train))
    response = await llm.acomplete(
        LLMRequest(
            model=model,
            messages=[Message(role="user", content=prompt)],
            temperature=GENERATION_TEMPERATURE,
        )
    )
    prompts = parse_string_list(response.content)[:n]
    return [Candidate(id=candidate_id(0, i), prompt=p, iteration=0) for i, p in enumerate(prompts)]
