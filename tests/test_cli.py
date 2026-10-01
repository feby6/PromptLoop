import json
from pathlib import Path
from typing import Any

import litellm
import pytest
from fakes import FakeCompletion
from typer.testing import CliRunner

from promptloop import cli
from promptloop.llm import LLMClient

runner = CliRunner()


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    root = tmp_path / "task"
    root.mkdir()
    (root / "task.yaml").write_text(
        "name: toy\ndescription: Say yes or no.\nscorer: exact\n"
        "target_model: groq/target\noptimizer_model: groq/opt\n"
    )
    train = [{"input": f"t{i}", "expected_output": "yes"} for i in range(10)]
    val = [{"input": "v0", "expected_output": "yes"}, {"input": "v1", "expected_output": "no"}]
    for name, rows in [("train.jsonl", train), ("val.jsonl", val)]:
        (root / name).write_text("\n".join(json.dumps(r) for r in rows))
    return root


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Route the CLI's LLMClient to a fake completion and a temp cache."""
    monkeypatch.setenv("PROMPTLOOP_CACHE_DIR", str(tmp_path / "cache"))

    def install(responder: Any) -> FakeCompletion:
        fake = FakeCompletion(responder=responder)
        monkeypatch.setattr(cli, "LLMClient", lambda s: LLMClient(s, completion_fn=fake))
        return fake

    return install


def test_baseline_reports_scores(task_dir: Path, fake_llm: Any) -> None:
    fake = fake_llm(lambda _: "yes")
    result = runner.invoke(cli.app, ["baseline", str(task_dir)])
    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 12
    assert fake.calls[0]["messages"][0]["content"] == "Say yes or no."
    assert "1.000" in result.output  # train
    assert "0.500" in result.output  # val
    assert "Val failures" in result.output


def test_baseline_exits_nonzero_on_call_errors(task_dir: Path, fake_llm: Any) -> None:
    fake_llm(lambda _: litellm.AuthenticationError("bad key", "groq", "groq/target"))
    result = runner.invoke(cli.app, ["baseline", str(task_dir)])
    assert result.exit_code == 1
    assert "12 call(s) failed" in result.output


def test_baseline_invalid_task(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["baseline", str(tmp_path)])
    assert result.exit_code == 1
    assert "Invalid task" in result.output


def test_run_optimises_and_writes_logs(
    task_dir: Path, fake_llm: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fakes import loop_responder

    # Settings are loaded inside the command, so point runs_dir via the config file.
    cfg = tmp_path / "providers.yaml"
    cfg.write_text(
        "defaults: {target_model: groq/target, optimizer_model: groq/opt}\n"
        f"runs_dir: {tmp_path / 'runs'}\n"
    )
    monkeypatch.setenv("PROMPTLOOP_CONFIG", str(cfg))
    fake_llm(loop_responder(lambda prompt, _: "yes" if prompt.startswith("v2") else "no"))

    result = runner.invoke(cli.app, ["run", str(task_dir), "--max-iterations", "3"])
    assert result.exit_code == 0, result.output
    assert "Best prompt" in result.output and "v2" in result.output
    report = next((tmp_path / "runs").glob("*-toy/report.md"))
    assert "optimised" in report.read_text()


def test_run_reports_failure(task_dir: Path, fake_llm: Any) -> None:
    fake_llm(lambda _: litellm.AuthenticationError("bad key", "groq", "groq/target"))
    result = runner.invoke(cli.app, ["run", str(task_dir), "--no-log"])
    assert result.exit_code == 1
    assert "Run failed" in result.output
