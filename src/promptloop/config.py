"""Settings: provider limits and defaults from config/providers.yaml, keys from .env."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "providers.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderLimits(_Strict):
    rpm: int | None = Field(default=None, gt=0)
    rpd: int | None = Field(default=None, gt=0)
    tpm: int | None = Field(default=None, gt=0)
    concurrency: int = Field(default=2, gt=0)


class ModelDefaults(_Strict):
    target_model: str
    optimizer_model: str


class RetrySettings(_Strict):
    max_retries: int = Field(default=5, ge=0)
    backoff_base_s: float = Field(default=2.0, gt=0)
    backoff_max_s: float = Field(default=60.0, gt=0)
    request_timeout_s: float = Field(default=60.0, gt=0)


class LoopSettings(_Strict):
    n_candidates: int = Field(default=4, gt=0)
    top_k: int = Field(default=2, gt=0)
    max_iterations: int = Field(default=5, gt=0)
    plateau_patience: int = Field(default=2, gt=0)
    target_score: float = Field(default=1.0, ge=0, le=1)


class Settings(_Strict):
    providers: dict[str, ProviderLimits] = Field(default_factory=dict)
    fallbacks: dict[str, list[str]] = Field(default_factory=dict)
    defaults: ModelDefaults
    retry: RetrySettings = Field(default_factory=RetrySettings)
    loop: LoopSettings = Field(default_factory=LoopSettings)
    cache_dir: Path = PROJECT_ROOT / ".cache"
    runs_dir: Path = PROJECT_ROOT / "runs"

    def limits_for(self, provider: str) -> ProviderLimits:
        """Limits for a provider, falling back to the `default` entry."""
        return self.providers.get(provider) or self.providers.get("default") or ProviderLimits()


def provider_of(model: str) -> str:
    """LiteLLM provider prefix of a model string: `groq/qwen/qwen3` -> `groq`."""
    return model.split("/", 1)[0] if "/" in model else "default"


def load_settings(config_path: Path | None = None, env_file: Path | None = None) -> Settings:
    """Load .env into the environment (for LiteLLM's API keys) and read the YAML config.

    `PROMPTLOOP_CONFIG` and `PROMPTLOOP_CACHE_DIR` override the default locations.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    path = config_path or Path(os.environ.get("PROMPTLOOP_CONFIG", DEFAULT_CONFIG_PATH))
    raw = yaml.safe_load(path.read_text()) or {}
    if cache_dir := os.environ.get("PROMPTLOOP_CACHE_DIR"):
        raw["cache_dir"] = cache_dir
    return Settings.model_validate(raw)
