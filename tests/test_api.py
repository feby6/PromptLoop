import time
from pathlib import Path
from typing import Any

import litellm
import pytest
from fakes import FakeCompletion, loop_responder
from fastapi.testclient import TestClient

from promptloop.api.app import create_app
from promptloop.config import Settings
from promptloop.llm import LLMClient, MemoryCache

USER_KEY = "sk-user-secret-key-123"
TASK = "Answer every question with yes or no."


def yes_after_refine(prompt: str, _input: str) -> str:
    return "yes" if prompt.startswith("v2") else "no"


def examples(n: int) -> list[dict[str, str]]:
    return [{"input": f"question {i}", "expected_output": "yes"} for i in range(n)]


def run_body(**overrides: Any) -> dict[str, Any]:
    return {
        "model": "groq/user-model",
        "api_key": USER_KEY,
        "task_description": TASK,
        "examples": examples(8),
        "max_iterations": 3,
        **overrides,
    }


@pytest.fixture
def fake() -> FakeCompletion:
    return FakeCompletion(responder=loop_responder(yes_after_refine))


@pytest.fixture
def client(settings: Settings, fake: FakeCompletion, tmp_path: Path):
    def factory(s: Settings, keys: dict[str, str]) -> LLMClient:
        # Same shape as the production factory, with the fake model plugged in.
        return LLMClient(
            s, cache=MemoryCache(), completion_fn=fake, api_keys=keys, allow_env_keys=False
        )

    app = create_app(settings, client_factory=factory, frontend_dist=tmp_path / "no-frontend")
    with TestClient(app) as c:
        yield c


def wait_for(client: TestClient, run_id: str, timeout_s: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = client.get(f"/api/runs/{run_id}").json()
        if status["status"] != "running":
            return status
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_health_and_config(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    config = client.get("/api/config").json()
    assert config["min_examples"] >= 2 and config["default_candidates"] == 3


def test_generate_examples_uses_the_users_key(client: TestClient, fake: FakeCompletion) -> None:
    res = client.post(
        "/api/examples/generate",
        json={
            "model": "groq/user-model",
            "api_key": USER_KEY,
            "task_description": TASK,
            "seed_examples": examples(1),
            "n": 3,
        },
    )
    assert res.status_code == 200, res.text
    assert res.json()["examples"] == [{"input": "new q", "expected_output": "yes"}]
    assert fake.calls[0]["api_key"] == USER_KEY


def test_generate_examples_upstream_failure_is_502_without_key(
    client: TestClient, fake: FakeCompletion
) -> None:
    fake.outcomes.append(litellm.AuthenticationError(f"bad {USER_KEY}", "groq", "groq/x"))
    res = client.post(
        "/api/examples/generate",
        json={
            "model": "groq/user-model",
            "api_key": USER_KEY,
            "task_description": TASK,
            "seed_examples": examples(1),
        },
    )
    assert res.status_code == 502
    assert USER_KEY not in res.text


def test_full_run_succeeds_and_streams_events(client: TestClient) -> None:
    res = client.post("/api/runs", json=run_body())
    assert res.status_code == 201, res.text
    created = res.json()
    assert created["scorer"] == "exact"  # auto-detected from short answers
    assert created["n_train"] + created["n_val"] == 8

    status = wait_for(client, created["run_id"])
    assert status["status"] == "succeeded", status["error"]
    result = status["result"]
    assert result["val_score"] == 1.0 and result["baseline_val_score"] == 0.0
    assert result["target_model"] == result["optimizer_model"] == "groq/user-model"
    assert status["lineage"][-1]["candidate"]["id"] == result["best_candidate"]["id"]

    # The stream replays everything and ends at the terminal event.
    with client.stream("GET", f"/api/runs/{created['run_id']}/events") as stream:
        body = "".join(stream.iter_text())
    assert body.count("event: iteration") == 2
    assert body.rstrip().splitlines()[-1].startswith("data:") and "event: result" in body

    # Resuming after an event id skips what the client already has.
    with client.stream(
        "GET", f"/api/runs/{created['run_id']}/events", headers={"Last-Event-ID": "1"}
    ) as stream:
        resumed = "".join(stream.iter_text())
    assert "id: 1\n" not in resumed and "id: 2\n" in resumed


def test_key_never_appears_in_responses(client: TestClient) -> None:
    run_id = client.post("/api/runs", json=run_body()).json()["run_id"]
    wait_for(client, run_id)
    assert USER_KEY not in client.get(f"/api/runs/{run_id}").text
    with client.stream("GET", f"/api/runs/{run_id}/events") as stream:
        assert USER_KEY not in "".join(stream.iter_text())


def test_failed_run_reports_redacted_error(client: TestClient, fake: FakeCompletion) -> None:
    def responder(kwargs: dict[str, Any]) -> Any:
        return litellm.AuthenticationError(f"invalid key {USER_KEY}", "groq", "groq/user-model")

    fake.responder = responder
    run_id = client.post("/api/runs", json=run_body()).json()["run_id"]
    status = wait_for(client, run_id)
    assert status["status"] == "failed"
    assert "invalid key ***" in status["error"] and USER_KEY not in status["error"]
    assert status["events"][-1]["type"] == "error"


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"examples": examples(3)}, "at least 6 examples"),
        ({"examples": examples(5) + examples(1)}, "duplicate inputs"),
        ({"model": "no-provider-prefix"}, "model"),
        ({"api_key": ""}, "api_key"),
        ({"task_description": "short"}, "task_description"),
    ],
)
def test_invalid_run_requests(client: TestClient, overrides: dict, fragment: str) -> None:
    res = client.post("/api/runs", json=run_body(**overrides))
    assert res.status_code == 422
    assert fragment in res.text


def test_unknown_run_is_404(client: TestClient) -> None:
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/events").status_code == 404
    assert client.delete("/api/runs/nope").status_code == 404


def test_busy_server_returns_429(settings: Settings, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("PROMPTLOOP_MAX_ACTIVE_RUNS", "0")
    app = create_app(settings, frontend_dist=tmp_path / "none")
    with TestClient(app) as c:
        res = c.post("/api/runs", json=run_body())
    assert res.status_code == 429


def test_cancel_run(settings: Settings, tmp_path: Path) -> None:
    import asyncio

    async def slow_completion(**kwargs: Any) -> Any:
        await asyncio.sleep(30)

    def factory(s: Settings, keys: dict[str, str]) -> LLMClient:
        return LLMClient(s, cache=MemoryCache(), completion_fn=slow_completion, api_keys=keys)

    app = create_app(settings, client_factory=factory, frontend_dist=tmp_path / "none")
    with TestClient(app) as c:
        run_id = c.post("/api/runs", json=run_body()).json()["run_id"]
        assert c.delete(f"/api/runs/{run_id}").status_code == 200
        status = wait_for(c, run_id)
    assert status["status"] == "cancelled"
    assert status["events"][-1]["type"] == "cancelled"


def test_serves_frontend_when_built(settings: Settings, tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<h1>PromptLoop UI</h1>")
    with TestClient(create_app(settings, frontend_dist=dist)) as c:
        assert "PromptLoop UI" in c.get("/").text
        assert c.get("/api/health").status_code == 200  # API still takes precedence
