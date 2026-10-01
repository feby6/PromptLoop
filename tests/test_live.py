"""Real provider calls. Excluded by default; run with `uv run pytest -m live`."""

import os
from pathlib import Path

import pytest
from diskcache import Cache

from promptloop.config import load_settings, provider_of
from promptloop.llm import LLMClient
from promptloop.models import LLMRequest, Message

pytestmark = pytest.mark.live

settings = load_settings()
KEY_ENV = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
MODELS = sorted({settings.defaults.target_model, settings.defaults.optimizer_model})


@pytest.mark.parametrize("model", MODELS)
def test_live_call_then_cache_hit(model: str, tmp_path: Path) -> None:
    key_env = KEY_ENV.get(provider_of(model))
    if key_env and not os.environ.get(key_env):
        pytest.skip(f"{key_env} not set")
    cache = Cache(str(tmp_path / "cache"))
    llm = LLMClient(settings, cache=cache)
    req = LLMRequest(
        model=model,
        messages=[Message(role="user", content="Reply with exactly one word: positive")],
        max_tokens=512,
    )
    first = llm.complete(req)
    second = llm.complete(req)
    llm.close()
    assert "positive" in first.content.lower()
    assert not first.cached and second.cached
    assert second.content == first.content
