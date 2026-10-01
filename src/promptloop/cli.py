"""Command-line interface for PromptLoop (development, benchmarks, and serving the app)."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from promptloop import __version__
from promptloop.config import Settings, load_settings
from promptloop.data import DataError, TaskBundle, load_task
from promptloop.llm import LLMClient, LLMError
from promptloop.models import Candidate, EvalResult, RunEvent, RunResult
from promptloop.optimizer.common import OptimizerError
from promptloop.optimizer.evaluate import score_candidate
from promptloop.optimizer.graph import RunContext, optimise
from promptloop.report import STOP_REASON_TEXT, RunLogger
from promptloop.scorers import scorer_for_task

app = typer.Typer(help="PromptLoop: automatic prompt optimisation.", no_args_is_help=True)
console = Console()

TaskDir = Annotated[Path, typer.Argument(help="Task folder with task.yaml, train/val JSONL.")]


@app.callback()
def main() -> None:
    """PromptLoop: automatic prompt optimisation."""


@app.command()
def version() -> None:
    """Print the installed PromptLoop version."""
    console.print(f"promptloop {__version__}")


def _load(task_dir: Path, settings: Settings) -> TaskBundle:
    try:
        return load_task(task_dir, settings)
    except DataError as e:
        console.print(f"[red]Invalid task:[/red] {escape(str(e))}")
        raise typer.Exit(1) from e


@app.command()
def baseline(
    task_dir: TaskDir,
    show_failures: Annotated[int, typer.Option(help="Val failures to print.")] = 5,
) -> None:
    """Score the unoptimised baseline prompt (the task description itself) on train and val."""
    settings = load_settings()
    bundle = _load(task_dir, settings)
    task = bundle.task
    candidate = Candidate(id="baseline", prompt=task.description)

    async def run() -> tuple[EvalResult, EvalResult]:
        llm = LLMClient(settings)
        scorer = scorer_for_task(task, llm)
        try:
            train, val = await asyncio.gather(
                score_candidate(llm, candidate, bundle.train, task.target_model, scorer),
                score_candidate(llm, candidate, bundle.val, task.target_model, scorer),
            )
            return train, val
        finally:
            llm.close()

    console.print(f"[bold]{task.name}[/bold] · target {task.target_model} · scorer {task.scorer}")
    console.print(f"Baseline prompt: [italic]{escape(candidate.prompt)}[/italic]\n")
    with console.status("Running target model..."):
        train, val = asyncio.run(run())

    table = Table("split", "examples", "score", "failures", "errors")
    for name, ev in [("train", train), ("val", val)]:
        table.add_row(
            name,
            str(len(ev.results)),
            f"{ev.mean_score:.3f}",
            str(len(ev.failures())),
            str(len(ev.errors())),
        )
    console.print(table)
    _print_failures(val, show_failures)

    errors = train.errors() + val.errors()
    if errors:
        console.print(
            f"[yellow]{len(errors)} call(s) failed, e.g.:[/yellow] {escape(errors[0].error or '')}"
        )
        raise typer.Exit(1)


@app.command()
def run(
    task_dir: TaskDir,
    candidates: Annotated[int | None, typer.Option(help="Candidates per round.")] = None,
    max_iterations: Annotated[int | None, typer.Option(help="Max scoring rounds.")] = None,
    baseline: Annotated[bool, typer.Option(help="Also score the raw task description.")] = True,
    log: Annotated[bool, typer.Option(help="Write the run to runs/.")] = True,
    show_failures: Annotated[int, typer.Option(help="Val failures to print.")] = 5,
) -> None:
    """Optimise the prompt for a task and report the best one."""
    settings = load_settings()
    bundle = _load(task_dir, settings)
    task = bundle.task
    overrides: dict[str, object] = {"include_baseline": baseline}
    if candidates is not None:
        overrides["n_candidates"] = candidates
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    loop = settings.loop.model_copy(update=overrides)
    logger = RunLogger(settings.runs_dir, task.name) if log else None

    console.print(
        f"[bold]{task.name}[/bold] · target {task.target_model} · optimiser "
        f"{task.optimizer_model} · scorer {task.scorer} · "
        f"{len(bundle.train)} train / {len(bundle.val)} val"
    )

    def on_event(event: RunEvent) -> None:
        style = "bold green" if event.type == "iteration" else "dim"
        console.print(f"[{style}]{escape(event.message)}[/{style}]")

    async def go() -> RunResult:
        llm = LLMClient(settings)
        try:
            ctx = RunContext(
                llm=llm,
                task=task,
                train=bundle.train,
                val=bundle.val,
                scorer=scorer_for_task(task, llm),
                loop=loop,
                on_event=on_event,
                logger=logger,
            )
            return await optimise(ctx)
        finally:
            llm.close()

    try:
        result = asyncio.run(go())
    except (LLMError, OptimizerError) as e:
        console.print(f"[red]Run failed:[/red] {escape(str(e))}")
        raise typer.Exit(1) from e
    _print_result(result, show_failures)
    if logger:
        console.print(f"\nRun log and report: [cyan]{logger.path}[/cyan]")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (dev).")] = False,
) -> None:
    """Start the web app (API + built frontend)."""
    import uvicorn  # imported lazily: only this command needs the server stack

    uvicorn.run("promptloop.api.app:create_app", factory=True, host=host, port=port, reload=reload)


def _print_result(result: RunResult, show_failures: int) -> None:
    def fmt(score: float | None) -> str:
        return "n/a" if score is None else f"{score:.3f}"

    table = Table("prompt", "train", "val (held-out)", title="Scores")
    table.add_row("baseline", fmt(result.baseline_train_score), fmt(result.baseline_val_score))
    table.add_row(
        f"optimised ({result.best_candidate.id})", fmt(result.train_score), fmt(result.val_score)
    )
    console.print(table)
    console.print(
        f"Stopped: {STOP_REASON_TEXT[result.stop_reason]} after {len(result.history)} round(s). "
        f"LLM calls: {result.usage.calls} (+{result.usage.cache_hits} cached)."
    )
    console.print(Panel(escape(result.best_candidate.prompt), title="Best prompt"))
    _print_failures(result.val_eval, show_failures)


def _print_failures(ev: EvalResult, limit: int) -> None:
    failures = [f for f in ev.failures() if not f.error][:limit]
    if not failures:
        return
    table = Table("input", "expected", "output", "score", title="Val failures", show_lines=True)
    for f in failures:
        table.add_row(_clip(f.input), _clip(f.expected_output), _clip(f.output), f"{f.score:.2f}")
    console.print(table)


def _clip(text: str, width: int = 70) -> str:
    text = " ".join(text.split())
    return escape(text if len(text) <= width else text[: width - 1] + "…")
