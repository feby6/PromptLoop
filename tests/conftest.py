from pathlib import Path

import pytest
from diskcache import Cache

from promptloop.config import ModelDefaults, ProviderLimits, RetrySettings, Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        providers={"groq": ProviderLimits(rpm=None, rpd=None, tpm=None, concurrency=2)},
        defaults=ModelDefaults(target_model="groq/target", optimizer_model="groq/optimizer"),
        retry=RetrySettings(max_retries=3, backoff_base_s=1.0, backoff_max_s=30.0),
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )


@pytest.fixture
def cache(tmp_path: Path):
    c = Cache(str(tmp_path / "llm-cache"))
    yield c
    c.close()
