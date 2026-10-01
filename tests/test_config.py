from pathlib import Path

import pytest
from pydantic import ValidationError

from promptloop.config import DEFAULT_CONFIG_PATH, ProviderLimits, load_settings, provider_of


def test_provider_of() -> None:
    assert provider_of("groq/qwen/qwen3.8-27b") == "groq"
    assert provider_of("gemini/gemini-flash") == "gemini"
    assert provider_of("gpt-4o") == "default"


def test_repo_config_loads(tmp_path: Path) -> None:
    settings = load_settings(DEFAULT_CONFIG_PATH, env_file=tmp_path / "missing.env")
    assert settings.defaults.target_model.startswith("groq/")
    # No up-front limits by default: users bring their own keys and tiers.
    assert settings.limits_for("groq").rpd is None


def test_limits_for_unknown_provider_uses_default(tmp_path: Path) -> None:
    cfg = tmp_path / "p.yaml"
    cfg.write_text(
        "providers:\n  default: {rpm: 7}\n"
        "defaults: {target_model: groq/a, optimizer_model: groq/b}\n"
    )
    settings = load_settings(cfg, env_file=tmp_path / "missing.env")
    assert settings.limits_for("anthropic").rpm == 7


def test_limits_for_without_default_entry(tmp_path: Path) -> None:
    cfg = tmp_path / "p.yaml"
    cfg.write_text("defaults: {target_model: groq/a, optimizer_model: groq/b}\n")
    settings = load_settings(cfg, env_file=tmp_path / "missing.env")
    assert settings.limits_for("groq") == ProviderLimits()


def test_cache_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPTLOOP_CACHE_DIR", str(tmp_path / "c"))
    settings = load_settings(DEFAULT_CONFIG_PATH, env_file=tmp_path / "missing.env")
    assert settings.cache_dir == tmp_path / "c"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "p.yaml"
    cfg.write_text("defaults: {target_model: a, optimizer_model: b}\ntypo_key: 1\n")
    with pytest.raises(ValidationError):
        load_settings(cfg, env_file=tmp_path / "missing.env")
