import json
from pathlib import Path

import pytest

from promptloop.config import PROJECT_ROOT, Settings
from promptloop.data import DataError, load_examples, load_task, split_examples
from promptloop.models import Example


def _write_task(
    root: Path, n_train: int = 10, n_val: int = 2, task_yaml: str | None = None
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "task.yaml").write_text(
        task_yaml
        or "name: t\ndescription: d\nscorer: exact\ntarget_model: groq/a\noptimizer_model: groq/b\n"
    )
    for name, n, prefix in [("train.jsonl", n_train, "tr"), ("val.jsonl", n_val, "va")]:
        lines = [json.dumps({"input": f"{prefix}{i}", "expected_output": "x"}) for i in range(n)]
        (root / name).write_text("\n".join(lines) + "\n")
    return root


def test_loads_valid_task(tmp_path: Path) -> None:
    bundle = load_task(_write_task(tmp_path / "t"))
    assert bundle.task.name == "t"
    assert len(bundle.train) == 10 and len(bundle.val) == 2


def test_example_task_in_repo_loads(settings: Settings) -> None:
    bundle = load_task(PROJECT_ROOT / "examples" / "tasks" / "sentiment", settings)
    assert bundle.task.target_model == settings.defaults.target_model
    assert len(bundle.train) >= 10


def test_models_default_from_settings(tmp_path: Path, settings: Settings) -> None:
    root = _write_task(tmp_path / "t", task_yaml="name: t\ndescription: d\nscorer: exact\n")
    bundle = load_task(root, settings)
    assert bundle.task.optimizer_model == "groq/optimizer"


def test_missing_models_without_settings_fails(tmp_path: Path) -> None:
    root = _write_task(tmp_path / "t", task_yaml="name: t\ndescription: d\nscorer: exact\n")
    with pytest.raises(DataError, match="target_model"):
        load_task(root)


def test_too_few_train_examples(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="at least 10"):
        load_task(_write_task(tmp_path / "t", n_train=9))


def test_empty_val(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="val.jsonl"):
        load_task(_write_task(tmp_path / "t", n_val=0))


def test_train_val_overlap_rejected(tmp_path: Path) -> None:
    root = _write_task(tmp_path / "t")
    with (root / "val.jsonl").open("a") as f:
        f.write(json.dumps({"input": "tr0", "expected_output": "x"}) + "\n")
    with pytest.raises(DataError, match="both train and val"):
        load_task(root)


def test_missing_task_yaml(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="task.yaml"):
        load_task(tmp_path)


def test_bad_jsonl_reports_line_number(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"input": "a", "expected_output": "b"}\n\n{not json}\n')
    with pytest.raises(DataError, match=r"train.jsonl:3"):
        load_examples(path)


def test_example_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text('{"input": "a"}\n')
    with pytest.raises(DataError, match=r"train.jsonl:1"):
        load_examples(path)


def test_json_match_task_requires_json_expected(tmp_path: Path) -> None:
    root = _write_task(
        tmp_path / "t",
        task_yaml="name: t\ndescription: d\nscorer: json_match\n"
        "target_model: groq/a\noptimizer_model: groq/b\n",
    )
    with pytest.raises(DataError, match="train.jsonl example 1: expected_output is not JSON"):
        load_task(root)


def _examples(n: int) -> list[Example]:
    return [Example(input=f"i{k}", expected_output="o") for k in range(n)]


def test_split_examples_sizes_and_determinism() -> None:
    train, val = split_examples(_examples(16), val_fraction=0.3, min_val=2)
    assert (len(train), len(val)) == (11, 5)
    assert {e.input for e in train}.isdisjoint({e.input for e in val})
    again_train, again_val = split_examples(_examples(16), val_fraction=0.3, min_val=2)
    assert again_train == train and again_val == val


def test_split_examples_min_val_and_keeps_train() -> None:
    assert [len(x) for x in split_examples(_examples(6), 0.1, min_val=2)] == [4, 2]
    assert [len(x) for x in split_examples(_examples(2), 0.5, min_val=5)] == [1, 1]
    with pytest.raises(DataError):
        split_examples(_examples(1), 0.3, min_val=1)
