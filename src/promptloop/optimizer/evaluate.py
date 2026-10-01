"""Score executed outputs and aggregate them per candidate."""

from promptloop.llm import LLMClient, LLMError, QuotaExhausted
from promptloop.models import Candidate, EvalResult, Example, ExampleOutput, ExampleResult
from promptloop.optimizer.common import gather_or_cancel
from promptloop.optimizer.execute import execute_candidate
from promptloop.scorers import AsyncScorer, Scorer
from promptloop.scorers.judge import JudgeError


async def score_candidate(
    llm: LLMClient,
    candidate: Candidate,
    examples: list[Example],
    model: str,
    scorer: Scorer | AsyncScorer,
) -> EvalResult:
    """Execute `candidate` on `examples` with `model`, then score the outputs."""
    outputs = await execute_candidate(llm, candidate, examples, model)
    return await evaluate(candidate.id, outputs, scorer)


async def evaluate(
    candidate_id: str, outputs: list[ExampleOutput], scorer: Scorer | AsyncScorer
) -> EvalResult:
    """Score each output. Failed target calls score 0 and keep their error.

    A judge that fails to grade an output also scores 0 with an error, so one bad
    judge reply can't sink a whole run, but the failure stays visible in the report.
    """
    results = await gather_or_cancel(*(_score_one(o, scorer) for o in outputs))
    return EvalResult(candidate_id=candidate_id, results=results)


async def _score_one(output: ExampleOutput, scorer: Scorer | AsyncScorer) -> ExampleResult:
    score, error = 0.0, output.error
    if error is None:
        try:
            if isinstance(scorer, AsyncScorer):
                score = await scorer.ascore(output.output, output.expected_output, output.input)
            else:
                score = scorer.score(output.output, output.expected_output)
        except QuotaExhausted:
            raise
        except (LLMError, JudgeError) as e:
            error = f"scoring failed: {e}"
    return ExampleResult(
        input=output.input,
        expected_output=output.expected_output,
        output=output.output,
        score=score,
        error=error,
    )
