import asyncio

import pytest
from diskcache import Cache
from fakes import FakeClock, FakeCompletion, loop_responder

from promptloop.config import LoopSettings, Settings
from promptloop.llm import LLMClient
from promptloop.models import Example, IterationRecord, RunEvent, Task
from promptloop.optimizer.common import OptimizerError
from promptloop.optimizer.graph import RunContext, decide_stop, optimise
from promptloop.report import RunLogger
from promptloop.scorers import ExactScorer

TASK = Task(
    name="toy",
    description="Answer the question.",
    scorer="exact",
    target_model="groq/target",
    optimizer_model="groq/opt",
)
TRAIN = [Example(input=f"train {i}", expected_output="yes") for i in range(4)]
VAL = [Example(input=f"val {i}", expected_output="yes") for i in range(3)]


def answers_yes_for_v2(prompt: str, _input: str) -> str:
    """Target model that only gets things right once the prompt has been refined."""
    return "yes" if prompt.startswith("v2") else "no"


def record(iteration: int, best: float) -> IterationRecord:
    return IterationRecord(
        iteration=iteration, candidates=[], evals=[], best_candidate_id="c", best_score=best
    )


@pytest.mark.parametrize(
    ("bests", "expected"),
    [
        ([0.5], None),
        ([1.0], "target_reached"),
        ([0.2, 0.4, 0.6, 0.8, 0.9], "max_iterations"),
        ([0.5, 0.5, 0.5], "plateau"),
        ([0.5, 0.5, 0.6], None),  # improved within the patience window
        ([0.5, 0.6, 0.6], None),  # only one stale round so far
    ],
)
def test_decide_stop(bests: list[float], expected: str | None) -> None:
    loop = LoopSettings(max_iterations=5, plateau_patience=2, target_score=1.0)
    history = [record(i, b) for i, b in enumerate(bests)]
    assert decide_stop(history, loop) == expected


def make_ctx(
    settings: Settings, cache: Cache, fake: FakeCompletion, **loop_overrides: object
) -> tuple[RunContext, list[RunEvent]]:
    clock = FakeClock()
    llm = LLMClient(settings, cache=cache, completion_fn=fake, clock=clock, sleep=clock.sleep)
    events: list[RunEvent] = []
    loop = LoopSettings(**{"n_candidates": 3, "top_k": 2, "max_iterations": 4, **loop_overrides})
    ctx = RunContext(
        llm=llm,
        task=TASK,
        train=TRAIN,
        val=VAL,
        scorer=ExactScorer(),
        loop=loop,
        on_event=events.append,
    )
    return ctx, events


def test_loop_improves_and_reports_against_baseline(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion(responder=loop_responder(answers_yes_for_v2))
    ctx, events = make_ctx(settings, cache, fake)
    result = asyncio.run(optimise(ctx))

    assert result.stop_reason == "target_reached"
    assert [r.best_score for r in result.history] == [0.0, 1.0]
    assert result.best_candidate.prompt.startswith("v2")
    assert result.best_candidate.parent_ids[0] == result.history[0].best_candidate_id
    assert (result.train_score, result.val_score) == (1.0, 1.0)
    assert (result.baseline_train_score, result.baseline_val_score) == (0.0, 0.0)
    assert (result.n_train, result.n_val) == (4, 3)
    # Round 1 scores the baseline plus 3 generated candidates.
    assert [c.id for c in result.history[0].candidates] == [
        "baseline",
        "it0-c0",
        "it0-c1",
        "it0-c2",
    ]
    assert result.history[1].critique == "The prompt never says to answer yes."
    assert result.usage.calls == len(fake.calls)

    types = [e.type for e in events]
    assert types[0] == "phase" and types[-1] == "result"
    assert types.count("iteration") == 2
    assert events[-1].data["lineage"][-1]["candidate"]["id"] == result.best_candidate.id


def test_val_examples_never_reach_the_optimiser(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion(responder=loop_responder(answers_yes_for_v2))
    ctx, _ = make_ctx(settings, cache, fake)
    asyncio.run(optimise(ctx))
    optimiser_prompts = [
        c["messages"][0]["content"] for c in fake.calls if c["messages"][0]["role"] == "user"
    ]
    assert optimiser_prompts
    assert not any(v.input in p for p in optimiser_prompts for v in VAL)


def test_plateau_stops_early(settings: Settings, cache: Cache) -> None:
    # The target never improves, so the loop should stop on plateau, not run all rounds.
    fake = FakeCompletion(responder=loop_responder(lambda p, i: "no"))
    ctx, _ = make_ctx(settings, cache, fake, max_iterations=8, plateau_patience=2)
    result = asyncio.run(optimise(ctx))
    assert result.stop_reason == "plateau"
    assert len(result.history) == 3
    # On a tie the earliest candidate (the baseline) wins.
    assert result.best_candidate.id == "baseline"


def test_without_baseline(settings: Settings, cache: Cache) -> None:
    fake = FakeCompletion(responder=loop_responder(answers_yes_for_v2))
    ctx, _ = make_ctx(settings, cache, fake, include_baseline=False)
    result = asyncio.run(optimise(ctx))
    assert result.baseline_val_score is None
    assert "baseline" not in [c.id for r in result.history for c in r.candidates]


def test_aborts_when_every_call_fails(settings: Settings, cache: Cache) -> None:
    import litellm

    def responder(kwargs: dict) -> object:
        if kwargs["messages"][0]["role"] == "system":
            return litellm.AuthenticationError("invalid api key", "groq", "groq/target")
        return loop_responder(answers_yes_for_v2)(kwargs)

    ctx, _ = make_ctx(settings, cache, FakeCompletion(responder=responder))
    with pytest.raises(OptimizerError, match="every call in this round failed"):
        asyncio.run(optimise(ctx))


def test_logger_writes_rounds_and_report(settings: Settings, cache: Cache, tmp_path) -> None:
    fake = FakeCompletion(responder=loop_responder(answers_yes_for_v2))
    ctx, _ = make_ctx(settings, cache, fake)
    ctx.logger = RunLogger(tmp_path, "toy task")
    asyncio.run(optimise(ctx))
    files = sorted(p.relative_to(ctx.logger.path).as_posix() for p in ctx.logger.path.rglob("*.*"))
    assert files == [
        "iterations/round_00.json",
        "iterations/round_01.json",
        "report.md",
        "result.json",
    ]
    assert ctx.logger.path.name.endswith("-toy_task")


def test_quota_exhaustion_aborts_instead_of_scoring_zero(settings: Settings, cache: Cache) -> None:
    from promptloop.config import ProviderLimits
    from promptloop.llm import QuotaExhausted

    # Enough daily quota to generate candidates and start scoring, not to finish.
    settings.providers["groq"] = ProviderLimits(rpd=5, concurrency=1)
    fake = FakeCompletion(responder=loop_responder(answers_yes_for_v2))
    ctx, events = make_ctx(settings, cache, fake)
    with pytest.raises(QuotaExhausted):
        asyncio.run(optimise(ctx))
    assert not any(e.type == "iteration" for e in events)  # no round was "scored"
