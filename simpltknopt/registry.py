"""Model registry: fetch from URL with ETag caching, local overlay merge."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import httpx
import yaml

from .config import Config, get_config
from .models.registry_models import ModelEntry, PricingInfo

CACHE_DIR = Path.home() / ".stko"
REGISTRY_CACHE_FILE = CACHE_DIR / "registry_cache.yaml"
REGISTRY_ETAG_FILE = CACHE_DIR / "registry_etag.json"

# Seed registry bundled alongside the package — used as ultimate fallback
SEED_REGISTRY = Path(__file__).parent.parent / "registry" / "models.yaml"


class RegistryFetchError(RuntimeError):
    pass


class ModelRegistry:
    """Loads model entries from the hosted registry with local override support."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or get_config()
        self._models: dict[str, ModelEntry] = {}
        self._loaded = False

    # ── Public interface ───────────────────────────────────────────────────────

    def load(self, force_refresh: bool = False) -> None:
        """Load registry: try network, fall back to cache, fall back to seed."""
        raw = self._fetch(force_refresh=force_refresh)
        self._models = self._parse(raw)
        self._apply_overrides()
        self._loaded = True

    def all_models(self) -> list[ModelEntry]:
        self._ensure_loaded()
        return list(self._models.values())

    def get(self, model_id: str) -> Optional[ModelEntry]:
        self._ensure_loaded()
        return self._models.get(model_id)

    def by_provider(self, provider: str) -> list[ModelEntry]:
        self._ensure_loaded()
        return [m for m in self._models.values() if m.provider == provider]

    def models_for_providers(self, providers: list[str]) -> list[ModelEntry]:
        self._ensure_loaded()
        return [m for m in self._models.values() if m.provider in providers]

    # ── Fetch logic ────────────────────────────────────────────────────────────

    def _fetch(self, force_refresh: bool = False) -> dict:
        """Return registry dict: network → cache → seed, in that order."""
        url = self._config.registry_url

        # Load current ETag metadata
        etag_data = self._load_etag_meta()
        etag = etag_data.get("etag") if not force_refresh else None

        try:
            raw = self._fetch_url(url, etag=etag)
            if raw is not None:
                return raw
            # 304 Not Modified — use cache
            cached = self._load_cache()
            if cached:
                return cached
        except (httpx.RequestError, httpx.HTTPStatusError, RegistryFetchError, ImportError, Exception):
            pass

        # Network unavailable — use cache
        cached = self._load_cache()
        if cached:
            return cached

        # Last resort: seed registry bundled with the package
        return self._load_seed()

    def _fetch_url(self, url: str, etag: Optional[str] = None) -> Optional[dict]:
        """Fetch registry from URL. Returns None on 304. Raises on other errors."""
        headers = {}
        if etag:
            headers["If-None-Match"] = etag

        with httpx.Client(timeout=10.0) as client:
            resp = client.get(url, headers=headers, follow_redirects=True)

        if resp.status_code == 304:
            return None

        resp.raise_for_status()

        data = yaml.safe_load(resp.text)
        if not isinstance(data, dict):
            raise RegistryFetchError(f"Registry at {url} returned unexpected format")

        # Cache it
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        REGISTRY_CACHE_FILE.write_text(resp.text)

        new_etag = resp.headers.get("etag") or resp.headers.get("ETag")
        if new_etag:
            REGISTRY_ETAG_FILE.write_text(json.dumps({"etag": new_etag}))

        return data

    def _load_cache(self) -> Optional[dict]:
        try:
            text = REGISTRY_CACHE_FILE.read_text()
            data = yaml.safe_load(text)
            return data if isinstance(data, dict) else None
        except (FileNotFoundError, yaml.YAMLError):
            return None

    def _load_seed(self) -> dict:
        try:
            data = yaml.safe_load(SEED_REGISTRY.read_text())
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, yaml.YAMLError):
            return {"models": []}

    def _load_etag_meta(self) -> dict:
        try:
            return json.loads(REGISTRY_ETAG_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    # ── Parsing & overlay ──────────────────────────────────────────────────────

    def _parse(self, raw: dict) -> dict[str, ModelEntry]:
        models: dict[str, ModelEntry] = {}
        for entry in raw.get("models", []):
            try:
                entry = dict(entry)  # copy before mutating
                pricing = PricingInfo(**entry.pop("pricing", {}))
                model = ModelEntry(pricing=pricing, **entry)
                models[model.id] = model
            except Exception:
                pass
        return models

    def _apply_overrides(self) -> None:
        """Merge developer overrides from stko.yaml on top of fetched registry."""
        for override in self._config.registry_overrides:
            model_id = override.get("id")
            if not model_id:
                continue
            if model_id in self._models:
                existing = self._models[model_id]
                caps_override = override.get("capabilities", {})
                new_caps = {**existing.capabilities, **caps_override}
                self._models[model_id] = existing.model_copy(
                    update={"capabilities": new_caps}
                )
            else:
                # New model entry
                try:
                    raw_copy = dict(override)
                    pricing_raw = raw_copy.pop("pricing", {})
                    pricing = PricingInfo(**pricing_raw)
                    model = ModelEntry(pricing=pricing, **raw_copy)
                    self._models[model.id] = model
                except Exception:
                    pass

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()


# ── Convenience singleton ──────────────────────────────────────────────────────

_registry: Optional[ModelRegistry] = None


def get_registry(config: Optional[Config] = None, force_refresh: bool = False) -> ModelRegistry:
    global _registry
    if _registry is None or force_refresh:
        _registry = ModelRegistry(config)
        _registry.load(force_refresh=force_refresh)
    return _registry
