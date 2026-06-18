"""Configuration loading with four-level precedence chain.

Precedence (highest → lowest):
  1. Per-SubTask fields (quality_threshold, verify, forced_model)
  2. Per-run kwargs passed to Router/Executor (threshold=, verify=)
  3. Project stko.yaml (in cwd or any parent directory)
  4. User preferences  ~/.stko/preferences.yaml
  5. Hard-coded defaults
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

# ── Hard-coded defaults ────────────────────────────────────────────────────────

DEFAULTS: dict[str, Any] = {
    "quality_threshold": 0.75,
    "verify": False,
    "max_escalations": 2,
    "decomposition_model": "claude-haiku-4-5-20251001",
    "judge_model": "auto",
    "interactive": True,
    "on_quality_failure": "continue",
    "registry_url": "https://registry.simpltknopt.dev/models.yaml",
    "enabled_providers": ["anthropic", "openai", "google", "deepseek", "nvidia"],
}

PREFS_PATH = Path.home() / ".stko" / "preferences.yaml"
STKO_YAML_NAME = "stko.yaml"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_yaml_safe(path: Path) -> dict[str, Any]:
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError:
        return {}


def _find_project_yaml() -> Optional[Path]:
    """Walk up from cwd looking for stko.yaml."""
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        p = candidate / STKO_YAML_NAME
        if p.exists():
            return p
    return None


def _expand_env(value: str) -> str:
    """Expand ${VAR} references in string values."""
    import re
    return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), value)


def _resolve_env_vars(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_env(obj)
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj


# ── Main config class ──────────────────────────────────────────────────────────

class Config:
    """Merged configuration from all sources."""

    def __init__(
        self,
        project_yaml_path: Optional[Path] = None,
        prefs_path: Optional[Path] = None,
    ) -> None:
        self._prefs_path = prefs_path or PREFS_PATH
        self._project_path = project_yaml_path or _find_project_yaml()

        prefs_raw = _load_yaml_safe(self._prefs_path)
        project_raw = _resolve_env_vars(_load_yaml_safe(self._project_path) if self._project_path else {})

        self._prefs = prefs_raw
        self._project = project_raw

        # Merge defaults dict
        self._defaults: dict[str, Any] = {
            **DEFAULTS,
            **self._prefs,
            **project_raw.get("defaults", {}),
        }

    # ── Accessors ──────────────────────────────────────────────────────────────

    @property
    def quality_threshold(self) -> float:
        return float(self._defaults.get("quality_threshold", DEFAULTS["quality_threshold"]))

    @property
    def verify(self) -> bool:
        return bool(self._defaults.get("verify", DEFAULTS["verify"]))

    @property
    def max_escalations(self) -> int:
        return int(self._defaults.get("max_escalations", DEFAULTS["max_escalations"]))

    @property
    def decomposition_model(self) -> str:
        return str(self._defaults.get("decomposition_model", DEFAULTS["decomposition_model"]))

    @property
    def judge_model(self) -> str:
        return str(self._defaults.get("judge_model", DEFAULTS["judge_model"]))

    @property
    def interactive(self) -> bool:
        return bool(self._defaults.get("interactive", DEFAULTS["interactive"]))

    @property
    def on_quality_failure(self) -> str:
        return str(self._defaults.get("on_quality_failure", DEFAULTS["on_quality_failure"]))

    @property
    def registry_url(self) -> str:
        return str(self._project.get("registry_url", DEFAULTS["registry_url"]))

    @property
    def enabled_providers(self) -> list[str]:
        return list(self._defaults.get("enabled_providers", DEFAULTS["enabled_providers"]))

    @property
    def registry_overrides(self) -> list[dict]:
        return list(self._project.get("registry_overrides", []))

    @property
    def api_keys(self) -> dict[str, str]:
        raw = self._project.get("api_keys", {})
        return {k: str(v) for k, v in raw.items() if v}

    def effective_threshold(self, run_threshold: Optional[float] = None) -> float:
        """Resolve threshold: run-level > config > hard default."""
        if run_threshold is not None:
            return run_threshold
        return self.quality_threshold

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_preference(self, key: str, value: Any) -> None:
        """Persist a user preference to ~/.stko/preferences.yaml."""
        self._prefs_path.parent.mkdir(parents=True, exist_ok=True)
        existing = _load_yaml_safe(self._prefs_path)
        existing[key] = value
        with open(self._prefs_path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False)
        self._prefs[key] = value
        self._defaults[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        return self._prefs.get(key, default)

    def has_been_initialized(self) -> bool:
        return self._prefs_path.exists()

    # ── API key helpers ────────────────────────────────────────────────────────

    def get_api_key(self, provider: str) -> Optional[str]:
        """Check config file first, then environment variables."""
        if provider in self.api_keys and self.api_keys[provider]:
            return self.api_keys[provider]
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "nvidia": "NVIDIA_API_KEY",
        }
        env_var = env_map.get(provider)
        return os.environ.get(env_var, "") if env_var else None

    def configured_providers(self) -> list[str]:
        """Return providers that are both enabled and have an API key configured."""
        return [p for p in self.enabled_providers if self.get_api_key(p)]


# ── Singleton accessor ─────────────────────────────────────────────────────────

_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    global _config
    if _config is None or reload:
        _config = Config()
    return _config
