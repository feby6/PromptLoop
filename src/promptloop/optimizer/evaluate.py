"""Score executed outputs and aggregate them per candidate."""

from promptloop.llm import LLMClient
from promptloop.models import Candidate, EvalResult, Example, ExampleOutput, ExampleResult
from promptloop.optimizer.execute import execute_candidate
from promptloop.scorers import Scorer


async def score_candidate(
    llm: LLMClient, candidate: Candidate, examples: list[Example], model: str, scorer: Scorer
) -> EvalResult:
    """Execute `candidate` on `examples` with `model`, then score the outputs."""
    outputs = await execute_candidate(llm, candidate, examples, model)
    return evaluate(candidate.id, outputs, scorer)


def evaluate(candidate_id: str, outputs: list[ExampleOutput], scorer: Scorer) -> EvalResult:
    """Score each output; failed calls score 0 and keep their error."""
    results = [
        ExampleResult(
            input=o.input,
            expected_output=o.expected_output,
            output=o.output,
            score=0.0 if o.error else scorer.score(o.output, o.expected_output),
            error=o.error,
        )
        for o in outputs
    ]
    return EvalResult(candidate_id=candidate_id, results=results)
