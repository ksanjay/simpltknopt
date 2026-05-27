"""Tests for ModelRegistry: parsing, overlay, offline fallback."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from simpltknopt.config import Config
from simpltknopt.models.registry_models import ModelEntry
from simpltknopt.models.task import TaskType
from simpltknopt.registry import ModelRegistry

SAMPLE_REGISTRY = {
    "version": "2026-05",
    "models": [
        {
            "id": "test-cheap",
            "display_name": "Cheap Model",
            "provider": "openai",
            "pricing": {"input_per_mtok": 0.10, "output_per_mtok": 0.30},
            "context_window": 128000,
            "capabilities": {
                "code_generation": 0.80,
                "classification": 0.85,
                "summarization": 0.88,
            },
        },
        {
            "id": "test-flagship",
            "display_name": "Flagship Model",
            "provider": "anthropic",
            "pricing": {"input_per_mtok": 15.00, "output_per_mtok": 75.00},
            "context_window": 200000,
            "capabilities": {
                "code_generation": 0.97,
                "classification": 0.97,
                "summarization": 0.96,
            },
        },
    ],
}


def _make_config(overrides=None) -> Config:
    cfg = MagicMock(spec=Config)
    cfg.registry_url = "https://example.com/models.yaml"
    cfg.registry_overrides = overrides or []
    cfg.enabled_providers = ["openai", "anthropic"]
    return cfg


class TestModelRegistryParsing:
    def test_parse_all_models(self):
        cfg = _make_config()
        reg = ModelRegistry(config=cfg)
        models = reg._parse(SAMPLE_REGISTRY)
        assert len(models) == 2
        assert "test-cheap" in models
        assert "test-flagship" in models

    def test_model_entry_fields(self):
        cfg = _make_config()
        reg = ModelRegistry(config=cfg)
        models = reg._parse(SAMPLE_REGISTRY)
        cheap = models["test-cheap"]
        assert cheap.display_name == "Cheap Model"
        assert cheap.pricing.input_per_mtok == 0.10
        assert cheap.context_window == 128000

    def test_capability_score(self):
        cfg = _make_config()
        reg = ModelRegistry(config=cfg)
        models = reg._parse(SAMPLE_REGISTRY)
        cheap = models["test-cheap"]
        assert cheap.capability_score(TaskType.CODE_GENERATION) == 0.80
        assert cheap.capability_score(TaskType.TRANSLATION) == 0.0  # not listed → 0

    def test_malformed_entry_skipped(self):
        bad_registry = {
            "models": [
                {"id": "bad", "display_name": "Bad"},  # missing required fields
                SAMPLE_REGISTRY["models"][0],
            ]
        }
        cfg = _make_config()
        reg = ModelRegistry(config=cfg)
        models = reg._parse(bad_registry)
        # Only the valid model parses
        assert len(models) == 1


class TestModelRegistryOverlay:
    def test_capability_override(self):
        cfg = _make_config(overrides=[
            {"id": "test-cheap", "capabilities": {"code_generation": 0.95}}
        ])
        reg = ModelRegistry(config=cfg)
        reg._models = reg._parse(SAMPLE_REGISTRY)
        reg._apply_overrides()
        assert reg._models["test-cheap"].capabilities["code_generation"] == 0.95
        # Other capabilities unchanged
        assert reg._models["test-cheap"].capabilities["classification"] == 0.85

    def test_new_model_from_override(self):
        cfg = _make_config(overrides=[
            {
                "id": "custom-model",
                "display_name": "Custom",
                "provider": "openai",
                "pricing": {"input_per_mtok": 0.05, "output_per_mtok": 0.15},
                "context_window": 32000,
                "capabilities": {"code_generation": 0.70},
            }
        ])
        reg = ModelRegistry(config=cfg)
        reg._models = reg._parse(SAMPLE_REGISTRY)
        reg._apply_overrides()
        assert "custom-model" in reg._models


class TestModelRegistryFallback:
    def test_offline_uses_cache(self, tmp_path):
        cache_file = tmp_path / "registry_cache.yaml"
        cache_file.write_text(yaml.dump(SAMPLE_REGISTRY))

        cfg = _make_config()
        reg = ModelRegistry(config=cfg)

        with patch.object(reg, "_fetch_url", side_effect=Exception("network down")):
            with patch("simpltknopt.registry.REGISTRY_CACHE_FILE", cache_file):
                result = reg._fetch()

        assert "models" in result
        assert len(result["models"]) == 2

    def test_uses_seed_when_no_cache(self, tmp_path):
        cfg = _make_config()
        reg = ModelRegistry(config=cfg)
        missing_cache = tmp_path / "missing.yaml"

        with patch.object(reg, "_fetch_url", side_effect=Exception("network down")):
            with patch("simpltknopt.registry.REGISTRY_CACHE_FILE", missing_cache):
                result = reg._fetch()

        # Falls back to seed — should have real models
        assert "models" in result
        assert len(result["models"]) > 0
