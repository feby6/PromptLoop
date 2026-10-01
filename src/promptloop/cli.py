"""Command-line interface for PromptLoop."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from promptloop import __version__
from promptloop.config import load_settings
from promptloop.data import DataError, load_task
from promptloop.llm import LLMClient
from promptloop.models import Candidate, EvalResult
from promptloop.optimizer.evaluate import score_candidate
from promptloop.scorers import get_scorer

app = typer.Typer(help="PromptLoop: automatic prompt optimisation.", no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """PromptLoop: automatic prompt optimisation."""


@app.command()
def version() -> None:
    """Print the installed PromptLoop version."""
    console.print(f"promptloop {__version__}")


@app.command()
def baseline(
    task_dir: Annotated[Path, typer.Argument(help="Task folder with task.yaml, train/val JSONL.")],
    show_failures: Annotated[int, typer.Option(help="Val failures to print.")] = 5,
) -> None:
    """Score the unoptimised baseline prompt (the task description itself) on train and val."""
    settings = load_settings()
    try:
        bundle = load_task(task_dir, settings)
    except DataError as e:
        console.print(f"[red]Invalid task:[/red] {e}")
        raise typer.Exit(1) from e
    task = bundle.task
    scorer = get_scorer(task.scorer)
    candidate = Candidate(id="baseline", prompt=task.description)

    async def run() -> tuple[EvalResult, EvalResult]:
        llm = LLMClient(settings)
        try:
            return await asyncio.gather(
                score_candidate(llm, candidate, bundle.train, task.target_model, scorer),
                score_candidate(llm, candidate, bundle.val, task.target_model, scorer),
            )
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
