"""Tests for Router: routing algorithm, cost tiebreaker, parallel-group detection."""
import pytest
from unittest.mock import MagicMock

from simpltknopt.config import Config
from simpltknopt.models.registry_models import ModelEntry, PricingInfo
from simpltknopt.models.task import RoutingPlan, SubTask, TaskType
from simpltknopt.registry import ModelRegistry
from simpltknopt.router import Router, RoutingError


def _make_entry(
    id: str,
    provider: str,
    input_price: float,
    output_price: float,
    cap: float,
    context: int = 128000,
) -> ModelEntry:
    caps = {t.value: cap for t in TaskType}
    return ModelEntry(
        id=id,
        display_name=id,
        provider=provider,
        pricing=PricingInfo(input_per_mtok=input_price, output_per_mtok=output_price),
        context_window=context,
        capabilities=caps,
    )


def _make_registry(entries: list[ModelEntry]) -> ModelRegistry:
    reg = MagicMock(spec=ModelRegistry)
    reg.all_models.return_value = entries
    reg.models_for_providers.return_value = entries
    reg.get.side_effect = lambda mid: next((e for e in entries if e.id == mid), None)
    return reg


def _make_config(threshold: float = 0.75, providers=None) -> Config:
    cfg = MagicMock(spec=Config)
    cfg.quality_threshold = threshold
    cfg.verify = False
    cfg.enabled_providers = providers or ["openai", "anthropic"]
    cfg.effective_threshold.side_effect = lambda run_t: run_t if run_t is not None else threshold
    return cfg


class TestRouterBasic:
    def _setup(self, threshold=0.75):
        entries = [
            _make_entry("cheap-model", "openai", 0.10, 0.40, 0.80),
            _make_entry("mid-model", "openai", 1.00, 4.00, 0.90),
            _make_entry("flagship-model", "anthropic", 15.00, 75.00, 0.97),
        ]
        reg = _make_registry(entries)
        cfg = _make_config(threshold)
        return Router(registry=reg, config=cfg)

    def test_assigns_cheapest_qualifying_model(self):
        router = self._setup(threshold=0.75)
        subtask = SubTask(
            description="Test task",
            task_types=[TaskType.CODE_GENERATION],
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        )
        plan = router.plan([subtask], task_summary="test")
        assert plan.decisions[0].assigned_model == "cheap-model"

    def test_threshold_filters_cheap_model(self):
        """At threshold 0.85, cheap model (0.80) should not be selected."""
        router = self._setup(threshold=0.85)
        subtask = SubTask(
            description="Test task",
            task_types=[TaskType.CODE_GENERATION],
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        )
        plan = router.plan([subtask], task_summary="test")
        assert plan.decisions[0].assigned_model == "mid-model"

    def test_forced_model_bypasses_routing(self):
        router = self._setup()
        subtask = SubTask(
            description="Test task",
            task_types=[TaskType.CODE_GENERATION],
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            forced_model="flagship-model",
        )
        plan = router.plan([subtask])
        assert plan.decisions[0].assigned_model == "flagship-model"
        assert plan.decisions[0].override_reason == "forced_model"

    def test_no_qualifying_model_raises(self):
        entries = [_make_entry("weak", "openai", 0.10, 0.40, 0.50)]
        reg = _make_registry(entries)
        cfg = _make_config(threshold=0.90)
        router = Router(registry=reg, config=cfg)
        subtask = SubTask(
            description="Test task",
            task_types=[TaskType.CODE_GENERATION],
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        )
        with pytest.raises(RoutingError):
            router.plan([subtask])

    def test_cost_tiebreaker_uses_quality(self):
        """When two models cost the same, pick the one with higher capability."""
        entries = [
            _make_entry("model-a", "openai", 1.00, 4.00, 0.85),
            _make_entry("model-b", "openai", 1.00, 4.00, 0.92),
        ]
        reg = _make_registry(entries)
        cfg = _make_config(threshold=0.80)
        router = Router(registry=reg, config=cfg)
        subtask = SubTask(
            description="Test",
            task_types=[TaskType.CODE_GENERATION],
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        )
        plan = router.plan([subtask])
        assert plan.decisions[0].assigned_model == "model-b"  # higher quality wins

    def test_context_window_filter(self):
        """Models with insufficient context window should be excluded."""
        entries = [
            _make_entry("small-ctx", "openai", 0.10, 0.40, 0.90, context=1000),
            _make_entry("large-ctx", "openai", 1.00, 4.00, 0.90, context=200000),
        ]
        reg = _make_registry(entries)
        cfg = _make_config(threshold=0.80)
        router = Router(registry=reg, config=cfg)
        subtask = SubTask(
            description="Test",
            task_types=[TaskType.CODE_GENERATION],
            estimated_input_tokens=5000,
            estimated_output_tokens=5000,
        )
        plan = router.plan([subtask])
        assert plan.decisions[0].assigned_model == "large-ctx"

    def test_per_run_threshold_override(self):
        router = self._setup(threshold=0.75)
        subtask = SubTask(
            description="Test",
            task_types=[TaskType.CODE_GENERATION],
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
        )
        # Apply strict run threshold
        plan = router.plan([subtask], run_threshold=0.95)
        assert plan.quality_threshold == 0.95

    def test_subtask_threshold_overrides_run(self):
        """SubTask.quality_threshold takes highest precedence."""
        router = self._setup(threshold=0.75)
        subtask = SubTask(
            description="Test",
            task_types=[TaskType.CODE_GENERATION],
            estimated_input_tokens=1000,
            estimated_output_tokens=500,
            quality_threshold=0.95,
        )
        plan = router.plan([subtask], run_threshold=0.70)
        # Decision for this sub-task used 0.95 threshold
        assert plan.decisions[0].capability_score >= 0.95


class TestParallelGroupDetection:
    def _router(self):
        entries = [_make_entry("m", "openai", 0.10, 0.40, 0.90)]
        reg = _make_registry(entries)
        cfg = _make_config()
        return Router(registry=reg, config=cfg)

    def test_independent_tasks_single_wave(self):
        router = self._router()
        subtasks = [
            SubTask(id="a", description="A", task_types=[TaskType.SUMMARIZATION]),
            SubTask(id="b", description="B", task_types=[TaskType.SUMMARIZATION]),
        ]
        plan = router.plan(subtasks)
        assert len(plan.parallel_groups) == 1
        assert set(plan.parallel_groups[0]) == {"a", "b"}

    def test_linear_chain_separate_waves(self):
        router = self._router()
        subtasks = [
            SubTask(id="a", description="A", task_types=[TaskType.SUMMARIZATION]),
            SubTask(id="b", description="B", task_types=[TaskType.SUMMARIZATION], depends_on=["a"]),
            SubTask(id="c", description="C", task_types=[TaskType.SUMMARIZATION], depends_on=["b"]),
        ]
        plan = router.plan(subtasks)
        assert len(plan.parallel_groups) == 3
        assert plan.parallel_groups[0] == ["a"]
        assert plan.parallel_groups[1] == ["b"]
        assert plan.parallel_groups[2] == ["c"]

    def test_diamond_dependency(self):
        """A → (B, C) → D: B and C in same wave."""
        router = self._router()
        subtasks = [
            SubTask(id="a", description="A", task_types=[TaskType.SUMMARIZATION]),
            SubTask(id="b", description="B", task_types=[TaskType.SUMMARIZATION], depends_on=["a"]),
            SubTask(id="c", description="C", task_types=[TaskType.SUMMARIZATION], depends_on=["a"]),
            SubTask(id="d", description="D", task_types=[TaskType.SUMMARIZATION], depends_on=["b", "c"]),
        ]
        plan = router.plan(subtasks)
        assert len(plan.parallel_groups) == 3
        assert plan.parallel_groups[0] == ["a"]
        assert set(plan.parallel_groups[1]) == {"b", "c"}
        assert plan.parallel_groups[2] == ["d"]

    def test_circular_dependency_raises(self):
        router = self._router()
        subtasks = [
            SubTask(id="a", description="A", task_types=[TaskType.SUMMARIZATION], depends_on=["b"]),
            SubTask(id="b", description="B", task_types=[TaskType.SUMMARIZATION], depends_on=["a"]),
        ]
        with pytest.raises(RoutingError, match="Circular dependency"):
            router.plan(subtasks)
