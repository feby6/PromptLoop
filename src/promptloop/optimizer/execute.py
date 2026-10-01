"""Run a candidate prompt on examples with the target model."""

import asyncio

from promptloop.llm import LLMClient, LLMError
from promptloop.models import Candidate, Example, ExampleOutput, LLMRequest, Message


def build_request(
    prompt: str, example: Example, model: str, temperature: float = 0.0
) -> LLMRequest:
    """The candidate prompt is the system message; the example input is the user message."""
    return LLMRequest(
        model=model,
        messages=[
            Message(role="system", content=prompt),
            Message(role="user", content=example.input),
        ],
        temperature=temperature,
    )


async def _run_one(llm: LLMClient, request: LLMRequest, example: Example) -> ExampleOutput:
    try:
        response = await llm.acomplete(request)
    except LLMError as e:
        return ExampleOutput(
            input=example.input, expected_output=example.expected_output, output="", error=str(e)
        )
    return ExampleOutput(
        input=example.input,
        expected_output=example.expected_output,
        output=response.content,
        cached=response.cached,
    )


async def execute_candidate(
    llm: LLMClient,
    candidate: Candidate,
    examples: list[Example],
    model: str,
    temperature: float = 0.0,
) -> list[ExampleOutput]:
    """Run `candidate` on every example concurrently (the client enforces rate limits).

    Outputs keep the order of `examples`. A failed call becomes an output with `error`
    set instead of aborting the whole batch.
    """
    return list(
        await asyncio.gather(
            *(
                _run_one(llm, build_request(candidate.prompt, ex, model, temperature), ex)
                for ex in examples
            )
        )
    )
