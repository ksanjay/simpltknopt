"""Tests for Verifier: deterministic checks and LLM judge path."""
import json
from unittest.mock import MagicMock, patch

import pytest

from simpltknopt.config import Config
from simpltknopt.models.registry_models import ModelEntry, PricingInfo
from simpltknopt.models.task import SubTask, TaskType
from simpltknopt.registry import ModelRegistry
from simpltknopt.verifier import Verifier, VerificationResult


def _make_config(judge="auto", threshold=0.75) -> Config:
    cfg = MagicMock(spec=Config)
    cfg.judge_model = judge
    cfg.quality_threshold = threshold
    cfg.decomposition_model = "test-model"
    cfg.enabled_providers = ["openai"]
    return cfg


def _make_registry() -> ModelRegistry:
    entry = ModelEntry(
        id="gpt-4o-mini",
        display_name="GPT-4o mini",
        provider="openai",
        pricing=PricingInfo(input_per_mtok=0.15, output_per_mtok=0.60),
        context_window=128000,
        capabilities={"classification": 0.88},
    )
    reg = MagicMock(spec=ModelRegistry)
    reg.all_models.return_value = [entry]
    reg.models_for_providers.return_value = [entry]
    return reg


class TestDeterministicChecks:
    def _verifier(self):
        return Verifier(registry=_make_registry(), config=_make_config())

    def test_valid_python_returns_none(self):
        v = self._verifier()
        subtask = SubTask(
            description="Write a function",
            task_types=[TaskType.CODE_GENERATION],
        )
        result = v._deterministic_check(subtask, "```python\ndef foo():\n    return 1\n```")
        assert result is None  # valid syntax → let LLM judge

    def test_invalid_python_fails(self):
        v = self._verifier()
        subtask = SubTask(
            description="Write a function",
            task_types=[TaskType.CODE_GENERATION],
        )
        result = v._deterministic_check(subtask, "```python\ndef foo(\n    return 1\n```")
        assert result is not None
        assert not result.passed
        assert result.score < 0.5

    def test_valid_json_returns_none(self):
        v = self._verifier()
        subtask = SubTask(description="Extract data", task_types=[TaskType.TOOL_USE])
        result = v._deterministic_check(subtask, '{"key": "value"}')
        assert result is None

    def test_invalid_json_fails(self):
        v = self._verifier()
        subtask = SubTask(description="Extract data", task_types=[TaskType.TOOL_USE])
        result = v._deterministic_check(subtask, "{key: value}")
        assert result is not None
        assert not result.passed

    def test_non_code_task_returns_none(self):
        v = self._verifier()
        subtask = SubTask(description="Summarize doc", task_types=[TaskType.SUMMARIZATION])
        result = v._deterministic_check(subtask, "Here is the summary.")
        assert result is None


class TestLLMJudge:
    def test_judge_parses_good_response(self):
        cfg = _make_config(judge="gpt-4o-mini")
        v = Verifier(registry=_make_registry(), config=cfg)
        subtask = SubTask(description="Summarize", task_types=[TaskType.SUMMARIZATION])

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"score": 0.92, "passed": true, "reason": "Comprehensive summary."}'

        with patch("litellm.completion", return_value=mock_resp):
            result = v._llm_judge(subtask, "Some output", threshold=0.75)

        assert result.score == 0.92
        assert result.passed is True
        assert "Comprehensive" in result.reason

    def test_judge_fails_below_threshold(self):
        cfg = _make_config(judge="gpt-4o-mini")
        v = Verifier(registry=_make_registry(), config=cfg)
        subtask = SubTask(description="Summarize", task_types=[TaskType.SUMMARIZATION])

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = '{"score": 0.55, "passed": false, "reason": "Missing key points."}'

        with patch("litellm.completion", return_value=mock_resp):
            result = v._llm_judge(subtask, "Partial output", threshold=0.75)

        assert not result.passed

    def test_judge_error_fails_open(self):
        """Verifier errors should not block the pipeline."""
        cfg = _make_config(judge="gpt-4o-mini")
        v = Verifier(registry=_make_registry(), config=cfg)
        subtask = SubTask(description="Summarize", task_types=[TaskType.SUMMARIZATION])

        with patch("litellm.completion", side_effect=Exception("API error")):
            result = v._llm_judge(subtask, "output", threshold=0.75)

        assert result.passed is True  # fail open

    def test_auto_judge_selects_cheapest(self):
        """Auto judge selection picks cheapest model with classification >= 0.80."""
        cfg = _make_config(judge="auto")
        v = Verifier(registry=_make_registry(), config=cfg)
        judge = v._resolve_judge_model()
        assert judge == "gpt-4o-mini"  # only model in mock registry
