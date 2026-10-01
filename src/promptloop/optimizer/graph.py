"""The optimisation loop as a LangGraph state machine.

    generate ─► evaluate ─┬─► critique ─► refine ─► evaluate ─► ...
                          └─► finalize ─► END

`evaluate` decides whether to stop (target reached, max rounds, or plateau). Only the
train split flows through generate/critique/refine; `finalize` is the single place the
val split is used, to score the winner (and the baseline) on examples the optimiser
never saw.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from promptloop.config import LoopSettings
from promptloop.llm import LLMClient
from promptloop.models import (
    Candidate,
    EvalResult,
    EventType,
    Example,
    IterationRecord,
    RunEvent,
    RunResult,
    ScoredCandidate,
    StopReason,
    Task,
)
from promptloop.optimizer.common import OptimizerError, gather_or_cancel
from promptloop.optimizer.critique import critique_candidate
from promptloop.optimizer.evaluate import score_candidate
from promptloop.optimizer.generate import generate_candidates
from promptloop.optimizer.refine import refine_candidates
from promptloop.report import RunLogger, build_lineage
from promptloop.scorers import AsyncScorer, Scorer

BASELINE_ID = "baseline"
# Score differences below this are noise, not improvement (for plateau detection).
IMPROVEMENT_EPS = 1e-9


@dataclass
class RunContext:
    """Everything the graph nodes need that isn't loop state.

    Kept out of the LangGraph state because the client and callbacks aren't data.
    """

    llm: LLMClient
    task: Task
    train: list[Example]
    val: list[Example]
    scorer: Scorer | AsyncScorer
    loop: LoopSettings
    on_event: Callable[[RunEvent], None] | None = None
    logger: RunLogger | None = None

    def emit(self, type: EventType, message: str, **data: Any) -> None:
        if self.on_event:
            self.on_event(RunEvent(type=type, message=message, data=data))


class LoopState(TypedDict, total=False):
    iteration: int  # index of the current scoring round, starting at 0
    pending: list[Candidate]  # generated, not yet scored
    pool: list[ScoredCandidate]  # every candidate scored so far, in scoring order
    history: list[IterationRecord]
    critique: str | None  # diagnosis that produced `pending`
    stop_reason: StopReason | None
    result: RunResult


def rank(pool: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """Best first. The sort is stable, so on a tie the earlier candidate wins: a refined
    prompt must actually beat its ancestors (or the baseline) to be reported as better."""
    return sorted(pool, key=lambda s: -s.score)


def decide_stop(history: list[IterationRecord], loop: LoopSettings) -> StopReason | None:
    latest = history[-1]
    if latest.best_score >= loop.target_score:
        return "target_reached"
    if len(history) >= loop.max_iterations:
        return "max_iterations"
    if len(history) > loop.plateau_patience:
        # best_score is cumulative, so "no improvement for N rounds" is one comparison.
        earlier = history[-1 - loop.plateau_patience].best_score
        if latest.best_score <= earlier + IMPROVEMENT_EPS:
            return "plateau"
    return None


def _check_not_all_errors(evals: list[EvalResult]) -> None:
    """Abort when every call in a round failed (bad key, unknown model, quota gone):
    carrying on would just burn more of the user's quota on guaranteed zeros."""
    results = [r for e in evals for r in e.results]
    if results and all(r.error for r in results):
        raise OptimizerError(f"every call in this round failed, e.g.: {results[0].error}")


def build_graph(ctx: RunContext) -> Any:
    task, loop = ctx.task, ctx.loop

    async def generate(state: LoopState) -> LoopState:
        ctx.emit("phase", f"Generating {loop.n_candidates} candidate prompts")
        candidates = await generate_candidates(
            ctx.llm, task.optimizer_model, task.description, ctx.train, loop.n_candidates
        )
        if loop.include_baseline:
            candidates.insert(0, Candidate(id=BASELINE_ID, prompt=task.description))
        ctx.emit(
            "candidates",
            f"{len(candidates)} candidates ready",
            iteration=0,
            candidates=[c.model_dump(mode="json") for c in candidates],
        )
        return {"pending": candidates, "iteration": 0}

    async def evaluate(state: LoopState) -> LoopState:
        iteration, pending = state["iteration"], state["pending"]
        ctx.emit(
            "phase",
            f"Round {iteration + 1}: scoring {len(pending)} prompt(s) on {len(ctx.train)} examples",
        )
        evals = list(
            await gather_or_cancel(
                *(
                    score_candidate(ctx.llm, c, ctx.train, task.target_model, ctx.scorer)
                    for c in pending
                )
            )
        )
        _check_not_all_errors(evals)
        scored = [
            ScoredCandidate(candidate=c, evaluation=e) for c, e in zip(pending, evals, strict=True)
        ]
        pool = [*state["pool"], *scored]
        best = rank(pool)[0]
        record = IterationRecord(
            iteration=iteration,
            candidates=pending,
            evals=evals,
            best_candidate_id=best.candidate.id,
            best_score=best.score,
            critique=state.get("critique"),
        )
        history = [*state["history"], record]
        stop = decide_stop(history, loop)
        if ctx.logger:
            ctx.logger.log_iteration(record)
        ctx.emit(
            "iteration",
            f"Round {iteration + 1}: best train score {best.score:.3f}",
            record=record.model_dump(mode="json"),
            usage=ctx.llm.usage.model_dump(),
        )
        return {"pool": pool, "history": history, "stop_reason": stop, "pending": []}

    def route(state: LoopState) -> str:
        return "finalize" if state.get("stop_reason") else "critique"

    async def critique(state: LoopState) -> LoopState:
        best = rank(state["pool"])[0]
        ctx.emit("phase", f"Diagnosing failures of the best prompt ({best.candidate.id})")
        text = await critique_candidate(ctx.llm, task.optimizer_model, task.description, best)
        return {"critique": text}

    async def refine(state: LoopState) -> LoopState:
        pool, nxt = state["pool"], state["iteration"] + 1
        parents = rank(pool)[: loop.top_k]
        ctx.emit("phase", f"Writing {loop.n_candidates} improved prompts")
        candidates = await refine_candidates(
            ctx.llm,
            task.optimizer_model,
            task.description,
            parents,
            state.get("critique") or "",
            ctx.train,
            loop.n_candidates,
            nxt,
        )
        # A prompt already scored would only re-score identically; skip it.
        seen = {s.candidate.prompt.strip() for s in pool}
        candidates = [c for c in candidates if c.prompt.strip() not in seen]
        ctx.emit(
            "candidates",
            f"{len(candidates)} new candidates",
            iteration=nxt,
            candidates=[c.model_dump(mode="json") for c in candidates],
        )
        return {"pending": candidates, "iteration": nxt}

    async def finalize(state: LoopState) -> LoopState:
        pool = state["pool"]
        best = rank(pool)[0]
        baseline = next((s for s in pool if s.candidate.id == BASELINE_ID), None)
        to_score = [best.candidate]
        if baseline and baseline is not best:
            to_score.append(baseline.candidate)
        ctx.emit("phase", f"Scoring the best prompt on {len(ctx.val)} held-out examples")
        val_evals = await gather_or_cancel(
            *(score_candidate(ctx.llm, c, ctx.val, task.target_model, ctx.scorer) for c in to_score)
        )
        best_val = val_evals[0]
        baseline_val = None
        if baseline:
            baseline_val = best_val if baseline is best else val_evals[1]
        result = RunResult(
            task_name=task.name,
            target_model=task.target_model,
            optimizer_model=task.optimizer_model,
            scorer=task.scorer,
            best_candidate=best.candidate,
            train_score=best.score,
            val_score=best_val.mean_score,
            val_eval=best_val,
            baseline_train_score=baseline.score if baseline else None,
            baseline_val_score=baseline_val.mean_score if baseline_val else None,
            stop_reason=state["stop_reason"] or "max_iterations",
            history=state["history"],
            n_train=len(ctx.train),
            n_val=len(ctx.val),
            usage=ctx.llm.usage.model_copy(),
        )
        if ctx.logger:
            ctx.logger.finish(result)
        return {"result": result}

    graph = StateGraph(LoopState)
    graph.add_node("generate", generate)
    graph.add_node("evaluate", evaluate)
    graph.add_node("critique", critique)
    graph.add_node("refine", refine)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_conditional_edges("evaluate", route, {"critique": "critique", "finalize": "finalize"})
    graph.add_edge("critique", "refine")
    graph.add_edge("refine", "evaluate")
    graph.add_edge("finalize", END)
    return graph.compile()


async def optimise(ctx: RunContext) -> RunResult:
    """Run the full loop and return the result (also emitted as a "result" event)."""
    graph = build_graph(ctx)
    initial: LoopState = {
        "iteration": 0,
        "pending": [],
        "pool": [],
        "history": [],
        "critique": None,
        "stop_reason": None,
    }
    # Each refinement round is three node visits (critique, refine, evaluate); LangGraph
    # aborts at the recursion limit, so size it from max_iterations with headroom.
    limit = 3 * ctx.loop.max_iterations + 10
    state = await graph.ainvoke(initial, config={"recursion_limit": limit})
    result: RunResult = state["result"]
    ctx.emit(
        "result",
        f"Done: val score {result.val_score:.3f}",
        result=result.model_dump(mode="json"),
        lineage=[step.model_dump(mode="json") for step in build_lineage(result)],
    )
    return result
