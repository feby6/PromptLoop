"""Generate extra examples from the user's seed examples.

Web users often give only one or two examples, which is not enough to score prompts
honestly (a prompt can be tuned to fit one case perfectly). The model writes more, and
the user reviews and edits them before a run, because a model can get its own
"expected outputs" wrong.
"""

from promptloop.llm import LLMClient
from promptloop.models import Example, LLMRequest, Message
from promptloop.optimizer.common import format_examples, parse_examples
from promptloop.prompts import render

# Diversity matters more than determinism when inventing data.
SYNTHESIS_TEMPERATURE = 0.8
# Show the model every seed (up to this many), since the seeds define the format.
MAX_SEEDS_IN_PROMPT = 10


async def synthesize_examples(
    llm: LLMClient,
    model: str,
    task_description: str,
    seeds: list[Example],
    n: int,
) -> list[Example]:
    """Up to `n` new examples, excluding any that repeat a seed's input."""
    prompt = render(
        "synthesize",
        n=n,
        task=task_description,
        examples=format_examples(seeds, limit=MAX_SEEDS_IN_PROMPT),
    )
    response = await llm.acomplete(
        LLMRequest(
            model=model,
            messages=[Message(role="user", content=prompt)],
            temperature=SYNTHESIS_TEMPERATURE,
        )
    )
    seen = {_key(s.input) for s in seeds}
    fresh = []
    for example in parse_examples(response.content):
        if _key(example.input) not in seen:
            seen.add(_key(example.input))
            fresh.append(example)
    return fresh[:n]


def _key(text: str) -> str:
    # Case/whitespace-insensitive, so trivially reworded duplicates are caught.
    return " ".join(text.lower().split())
