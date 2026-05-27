"""Tests for CLI commands using Typer's test runner."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from simpltknopt.cli import app

runner = CliRunner()


def _mock_plan():
    from simpltknopt.models.task import RoutingPlan, SubTask, RoutingDecision, TaskType
    subtask = SubTask(
        id="test-task",
        description="Test task",
        task_types=[TaskType.CODE_GENERATION],
        estimated_input_tokens=500,
        estimated_output_tokens=300,
    )
    decision = RoutingDecision(
        subtask_id="test-task",
        assigned_model="gpt-4o-mini",
        assigned_model_display="GPT-4o mini",
        capability_score=0.88,
        estimated_cost_usd=0.0002,
        ranked_alternatives=[],
        parallel_group=0,
    )
    return RoutingPlan(
        task_summary="Test",
        subtasks=[subtask],
        decisions=[decision],
        total_estimated_cost_usd=0.0002,
        comparison={"Claude Opus 4.6": 0.05},
        parallel_groups=[["test-task"]],
        quality_threshold=0.75,
    )


class TestDoctorCommand:
    def test_doctor_runs(self):
        with patch("simpltknopt.config.get_config") as mock_cfg, \
             patch("simpltknopt.registry.get_registry") as mock_reg:
            cfg = MagicMock()
            cfg.get_api_key.return_value = "sk-test12345"
            cfg.quality_threshold = 0.75
            cfg.configured_providers.return_value = ["anthropic"]
            cfg.enabled_providers = ["anthropic"]
            mock_cfg.return_value = cfg

            reg = MagicMock()
            reg.all_models.return_value = [MagicMock()] * 10
            mock_reg.return_value = reg

            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0


class TestModelsListCommand:
    def test_models_list_runs(self):
        with patch("simpltknopt.registry.get_registry") as mock_reg:
            from simpltknopt.models.registry_models import ModelEntry, PricingInfo
            entry = ModelEntry(
                id="test-model",
                display_name="Test Model",
                provider="openai",
                pricing=PricingInfo(input_per_mtok=0.10, output_per_mtok=0.30),
                context_window=128000,
                capabilities={"code_generation": 0.80},
            )
            reg = MagicMock()
            reg.all_models.return_value = [entry]
            mock_reg.return_value = reg

            result = runner.invoke(app, ["models", "list"])
            assert result.exit_code == 0
            assert "Test Model" in result.output

    def test_models_list_by_task_type(self):
        with patch("simpltknopt.registry.get_registry") as mock_reg:
            from simpltknopt.models.registry_models import ModelEntry, PricingInfo
            entry = ModelEntry(
                id="test",
                display_name="Test",
                provider="openai",
                pricing=PricingInfo(input_per_mtok=0.10, output_per_mtok=0.30),
                context_window=128000,
                capabilities={"code_generation": 0.80},
            )
            reg = MagicMock()
            reg.all_models.return_value = [entry]
            mock_reg.return_value = reg

            result = runner.invoke(app, ["models", "list", "--task-type", "code_generation"])
            assert result.exit_code == 0


class TestConfigCommands:
    def test_config_show(self):
        with patch("simpltknopt.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.quality_threshold = 0.75
            cfg.verify = False
            cfg.max_escalations = 2
            cfg.decomposition_model = "claude-haiku-4-5-20251001"
            cfg.judge_model = "auto"
            cfg.interactive = True
            cfg.on_quality_failure = "continue"
            cfg.registry_url = "https://example.com"
            cfg.enabled_providers = ["anthropic"]
            cfg._prefs = {}
            cfg._project = {}
            mock_cfg.return_value = cfg

            result = runner.invoke(app, ["config", "show"])
            assert result.exit_code == 0

    def test_config_set(self, tmp_path):
        prefs_file = tmp_path / "preferences.yaml"
        with patch("simpltknopt.config.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.save_preference = MagicMock()
            mock_cfg.return_value = cfg

            result = runner.invoke(app, ["config", "set", "quality_threshold", "0.85"])
            assert result.exit_code == 0
            cfg.save_preference.assert_called_once_with("quality_threshold", 0.85)


class TestCostEstimateCommand:
    def test_cost_estimate_runs(self):
        with patch("simpltknopt.registry.get_registry") as mock_reg:
            from simpltknopt.models.registry_models import ModelEntry, PricingInfo
            entry = ModelEntry(
                id="test",
                display_name="Test",
                provider="openai",
                pricing=PricingInfo(input_per_mtok=0.10, output_per_mtok=0.30),
                context_window=128000,
                capabilities={},
            )
            reg = MagicMock()
            reg.all_models.return_value = [entry]
            mock_reg.return_value = reg

            result = runner.invoke(app, ["cost-estimate", "--input-tokens", "1000", "--output-tokens", "500"])
            assert result.exit_code == 0
