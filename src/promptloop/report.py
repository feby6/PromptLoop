"""Run reporting: prompt lineage with diffs, a markdown report, and on-disk run logs."""

import difflib
import json
import re
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from promptloop.models import Candidate, IterationRecord, RunResult

STOP_REASON_TEXT = {
    "target_reached": "reached the target score",
    "max_iterations": "hit the maximum number of rounds",
    "plateau": "stopped improving",
}
# Words and the whitespace between them, so a diff can be re-joined losslessly.
_TOKEN_RE = re.compile(r"\s+|\S+")


class DiffChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: Literal["equal", "insert", "delete"]
    text: str


class LineageStep(BaseModel):
    """One prompt on the path from a root candidate to the winner."""

    model_config = ConfigDict(extra="forbid")
    candidate: Candidate
    train_score: float | None
    diff_from_parent: list[DiffChunk]  # empty for the root


def word_diff(old: str, new: str) -> list[DiffChunk]:
    """Word-level diff. Prompts are often one long paragraph, where a line diff would
    just show "everything changed"."""
    a, b = _TOKEN_RE.findall(old), _TOKEN_RE.findall(new)
    chunks: list[DiffChunk] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            chunks.append(DiffChunk(op="equal", text="".join(a[i1:i2])))
            continue
        if op in ("delete", "replace"):
            chunks.append(DiffChunk(op="delete", text="".join(a[i1:i2])))
        if op in ("insert", "replace"):
            chunks.append(DiffChunk(op="insert", text="".join(b[j1:j2])))
    return chunks


def unified_diff(old: str, new: str, old_name: str = "before", new_name: str = "after") -> str:
    return "\n".join(
        difflib.unified_diff(old.splitlines(), new.splitlines(), old_name, new_name, lineterm="")
    )


def _scores_by_id(history: list[IterationRecord]) -> dict[str, float]:
    return {e.candidate_id: e.mean_score for record in history for e in record.evals}


def lineage(result: RunResult) -> list[Candidate]:
    """Root-to-winner chain, following each candidate's first (best) parent."""
    by_id = {c.id: c for record in result.history for c in record.candidates}
    chain = [result.best_candidate]
    while chain[-1].parent_ids and chain[-1].parent_ids[0] in by_id:
        chain.append(by_id[chain[-1].parent_ids[0]])
        if len(chain) > len(by_id):  # defensive: malformed history with a cycle
            break
    return chain[::-1]


def build_lineage(result: RunResult) -> list[LineageStep]:
    scores = _scores_by_id(result.history)
    chain = lineage(result)
    steps = []
    for i, candidate in enumerate(chain):
        diff = word_diff(chain[i - 1].prompt, candidate.prompt) if i else []
        steps.append(
            LineageStep(
                candidate=candidate, train_score=scores.get(candidate.id), diff_from_parent=diff
            )
        )
    return steps


def _one_line(text: str) -> str:
    # Multi-line model output would break markdown list items and inline code spans.
    return " ".join(text.split())


def _fmt(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.3f}"


def render_markdown(result: RunResult) -> str:
    """Human-readable report for a finished run (written next to the JSON logs)."""
    lines = [
        f"# PromptLoop report: {result.task_name}",
        "",
        f"- Target model: `{result.target_model}`",
        f"- Optimiser model: `{result.optimizer_model}`",
        f"- Scorer: `{result.scorer}`",
        f"- Examples: {result.n_train} train / {result.n_val} held-out val",
        f"- Stopped because it {STOP_REASON_TEXT[result.stop_reason]} "
        f"after {len(result.history)} round(s)",
        f"- LLM calls: {result.usage.calls} (+{result.usage.cache_hits} cache hits), "
        f"{result.usage.prompt_tokens + result.usage.completion_tokens} tokens",
        "",
        "## Scores",
        "",
        "| prompt | train | val (held-out) |",
        "|---|---|---|",
        f"| baseline (task description) | {_fmt(result.baseline_train_score)} "
        f"| {_fmt(result.baseline_val_score)} |",
        f"| optimised (`{result.best_candidate.id}`) | {_fmt(result.train_score)} "
        f"| {_fmt(result.val_score)} |",
        "",
        "## Best train score per round",
        "",
        "| round | candidates | best so far |",
        "|---|---|---|",
        *(
            f"| {r.iteration + 1} | {len(r.candidates)} | {r.best_score:.3f} |"
            for r in result.history
        ),
        "",
        "## Best prompt",
        "",
        "```text",
        result.best_candidate.prompt,
        "```",
        "",
    ]
    chain = lineage(result)
    if len(chain) > 1:
        lines += ["## How the prompt evolved", ""]
        for parent, child in pairwise(chain):
            lines += [
                f"### {parent.id} → {child.id}",
                "",
                "```diff",
                unified_diff(parent.prompt, child.prompt, parent.id, child.id),
                "```",
                "",
            ]
    failures = result.val_failures
    lines += ["## Held-out cases still failing", ""]
    if not failures:
        lines.append("None.")
    for f in failures:
        lines += [
            f"- **Input:** {_one_line(f.input)}",
            f"  - expected: `{_one_line(f.expected_output)}`",
            f"  - got: `{_one_line(f.output)}`" + (f" (error: {f.error})" if f.error else ""),
        ]
    return "\n".join(lines) + "\n"


class RunLogger:
    """Writes a run to `runs/<timestamp>-<task>/` as it progresses.

    Iterations are written as they finish, so a crashed or cancelled run still leaves a
    useful trail. CLI only: web runs don't write user data to disk.
    """

    def __init__(self, runs_dir: Path, task_name: str, now: datetime | None = None) -> None:
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        safe_name = re.sub(r"[^\w.-]+", "_", task_name)
        self.path = runs_dir / f"{stamp}-{safe_name}"
        (self.path / "iterations").mkdir(parents=True, exist_ok=True)

    def log_iteration(self, record: IterationRecord) -> None:
        self._write(f"iterations/round_{record.iteration:02d}.json", record.model_dump(mode="json"))

    def finish(self, result: RunResult) -> None:
        self._write("result.json", result.model_dump(mode="json"))
        (self.path / "report.md").write_text(render_markdown(result), encoding="utf-8")

    def _write(self, name: str, data: object) -> None:
        (self.path / name).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
