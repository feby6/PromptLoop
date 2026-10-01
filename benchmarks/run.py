"""Run the optimiser on each benchmark task and write benchmarks/RESULTS.md.

    uv run python benchmarks/run.py                 # all tasks in benchmarks/tasks
    uv run python benchmarks/run.py gsm8k ag_news   # selected tasks

Uses the default target/optimiser models from config/providers.yaml and keys from .env.
LLM calls are cached on disk, so re-running only pays for what changed.
"""

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

from promptloop.config import load_settings
from promptloop.data import load_task
from promptloop.llm import LLMClient
from promptloop.models import RunResult
from promptloop.optimizer.graph import RunContext, optimise
from promptloop.report import RunLogger
from promptloop.scorers import scorer_for_task

ROOT = Path(__file__).parent
TASKS_DIR = ROOT / "tasks"
RESULTS_DIR = ROOT / "results"
EXTRA_TASKS = [ROOT.parent / "examples" / "tasks" / "invoice_extraction"]


async def run_task(task_dir: Path) -> RunResult:
    settings = load_settings()
    bundle = load_task(task_dir, settings)
    llm = LLMClient(settings)
    try:
        ctx = RunContext(
            llm=llm,
            task=bundle.task,
            train=bundle.train,
            val=bundle.val,
            scorer=scorer_for_task(bundle.task, llm),
            loop=settings.loop,
            on_event=lambda e: print(f"  [{bundle.task.name}] {e.message}", flush=True),
            logger=RunLogger(settings.runs_dir, f"bench-{bundle.task.name}"),
        )
        return await optimise(ctx)
    finally:
        llm.close()


def fmt(score: float | None) -> str:
    return "n/a" if score is None else f"{score * 100:.0f}%"


def write_results_table() -> None:
    results = [
        RunResult.model_validate_json(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))
    ]
    lines = [
        "# Benchmark results",
        "",
        f"_Last updated {date.today().isoformat()}. Held-out (val) scores: the optimiser "
        "never sees these examples. Baseline = the task description used as the prompt._",
        "",
        "| task | model | scorer | train / val | baseline val | optimised val | Δ | rounds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        delta = (
            "n/a"
            if r.baseline_val_score is None
            else f"{(r.val_score - r.baseline_val_score) * 100:+.0f} pts"
        )
        lines.append(
            f"| {r.task_name} | `{r.target_model}` | {r.scorer} | {r.n_train} / {r.n_val} "
            f"| {fmt(r.baseline_val_score)} | {fmt(r.val_score)} | {delta} | {len(r.history)} |"
        )
    lines += [
        "",
        "Small val sets (a dozen examples) mean each example is worth ~8 points: treat "
        "differences of one or two examples as noise.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(names: list[str]) -> None:
    task_dirs = sorted(p for p in TASKS_DIR.iterdir() if p.is_dir()) + EXTRA_TASKS
    if names:
        task_dirs = [p for p in task_dirs if p.name in names]
    RESULTS_DIR.mkdir(exist_ok=True)
    for task_dir in task_dirs:
        print(f"== {task_dir.name}", flush=True)
        result = asyncio.run(run_task(task_dir))
        (RESULTS_DIR / f"{task_dir.name}.json").write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False)
        )
        print(
            f"  baseline val {fmt(result.baseline_val_score)} -> optimised {fmt(result.val_score)}"
        )
    write_results_table()


if __name__ == "__main__":
    main(sys.argv[1:])
