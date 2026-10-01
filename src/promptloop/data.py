"""Load a task directory: task.yaml + train.jsonl + val.jsonl."""

import json
import random
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from promptloop.config import Settings
from promptloop.models import Example, Task

MIN_TRAIN_EXAMPLES = 10


class DataError(ValueError):
    """A task directory is missing files or contains invalid data."""


class TaskBundle(BaseModel):
    """A task with its examples. `val` must never be shown to the optimiser."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task: Task
    train: list[Example]
    val: list[Example]
    path: Path


def load_examples(path: Path) -> list[Example]:
    """Parse a JSONL file of {"input", "expected_output"} objects; blank lines are skipped."""
    if not path.is_file():
        raise DataError(f"missing file: {path}")
    examples = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            examples.append(Example.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as e:
            raise DataError(f"{path}:{lineno}: invalid example: {e}") from e
    return examples


def load_task(task_dir: Path, settings: Settings | None = None) -> TaskBundle:
    """Load and validate a task directory.

    `target_model` / `optimizer_model` may be omitted from task.yaml when `settings`
    is given; the configured defaults are used instead.
    """
    task_dir = Path(task_dir)
    task_file = task_dir / "task.yaml"
    if not task_file.is_file():
        raise DataError(f"missing file: {task_file}")
    try:
        raw = yaml.safe_load(task_file.read_text())
    except yaml.YAMLError as e:
        raise DataError(f"{task_file}: invalid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise DataError(f"{task_file}: expected a mapping")
    if settings:
        raw.setdefault("target_model", settings.defaults.target_model)
        raw.setdefault("optimizer_model", settings.defaults.optimizer_model)
    try:
        task = Task.model_validate(raw)
    except ValidationError as e:
        raise DataError(f"{task_file}: {e}") from e

    train = load_examples(task_dir / "train.jsonl")
    val = load_examples(task_dir / "val.jsonl")
    if len(train) < MIN_TRAIN_EXAMPLES:
        raise DataError(f"need at least {MIN_TRAIN_EXAMPLES} train examples, got {len(train)}")
    if not val:
        raise DataError("val.jsonl has no examples")
    leaked = {e.input for e in train} & {e.input for e in val}
    if leaked:
        raise DataError(f"{len(leaked)} input(s) appear in both train and val")
    if task.scorer == "json_match":
        _check_expected_json(train, "train.jsonl")
        _check_expected_json(val, "val.jsonl")
    return TaskBundle(task=task, train=train, val=val, path=task_dir)


def _check_expected_json(examples: list[Example], filename: str) -> None:
    for i, example in enumerate(examples, start=1):
        try:
            json.loads(example.expected_output)
        except json.JSONDecodeError as e:
            raise DataError(f"{filename} example {i}: expected_output is not JSON: {e}") from e


def split_examples(
    examples: list[Example], val_fraction: float, min_val: int, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Shuffle deterministically and hold out a val split (web runs, where the user
    supplies one pool of examples). The fixed seed makes reruns comparable and cacheable.

    At least one example always stays in train, so the optimiser has something to learn
    from even when the pool is tiny.
    """
    if len(examples) < 2:
        raise DataError("need at least 2 examples to hold some out for validation")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_val = max(min_val, round(len(shuffled) * val_fraction))
    n_val = min(n_val, len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]
