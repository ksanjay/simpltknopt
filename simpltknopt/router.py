"""Routing engine: assign cheapest capable model per sub-task + parallel-group detection."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from .config import Config, get_config
from .models.registry_models import ModelEntry
from .models.task import RoutingDecision, RoutingPlan, SubTask, TaskType
from .registry import ModelRegistry, get_registry


class RoutingError(RuntimeError):
    pass


# Models to include in cost comparison (one per major provider/tier)
_COMPARISON_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "gpt-4o",
    "gemini/gemini-1.5-pro",
    "deepseek/deepseek-chat",
]


class Router:
    """Routes sub-tasks to models and detects parallel execution groups."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        config: Optional[Config] = None,
    ) -> None:
        self._registry = registry or get_registry()
        self._config = config or get_config()

    # ── Public interface ───────────────────────────────────────────────────────

    def plan(
        self,
        subtasks: list[SubTask],
        task_summary: str = "",
        run_threshold: Optional[float] = None,
        run_verify: Optional[bool] = None,
    ) -> RoutingPlan:
        """Build a full RoutingPlan for a list of sub-tasks."""
        effective_threshold = self._config.effective_threshold(run_threshold)
        effective_verify = run_verify if run_verify is not None else self._config.verify

        decisions: list[RoutingDecision] = []
        for subtask in subtasks:
            decision = self._route_subtask(subtask, effective_threshold, effective_verify)
            decisions.append(decision)

        parallel_groups = self._compute_parallel_groups(subtasks)
        self._annotate_groups(decisions, parallel_groups)

        total_cost = sum(d.estimated_cost_usd for d in decisions)
        comparison = self._build_comparison(subtasks)

        return RoutingPlan(
            task_summary=task_summary,
            subtasks=subtasks,
            decisions=decisions,
            total_estimated_cost_usd=total_cost,
            comparison=comparison,
            parallel_groups=parallel_groups,
            quality_threshold=effective_threshold,
            verify_enabled=effective_verify,
        )

    # ── Core routing algorithm ─────────────────────────────────────────────────

    def _route_subtask(
        self,
        subtask: SubTask,
        default_threshold: float,
        default_verify: bool,
    ) -> RoutingDecision:
        # 1. Resolve effective threshold (per-subtask overrides run/config)
        threshold = (
            subtask.quality_threshold
            if subtask.quality_threshold is not None
            else default_threshold
        )
        verify = subtask.verify or default_verify

        # 2. Forced model short-circuit
        if subtask.forced_model:
            entry = self._registry.get(subtask.forced_model)
            cost = (
                entry.estimated_cost(
                    subtask.estimated_input_tokens, subtask.estimated_output_tokens
                )
                if entry
                else 0.0
            )
            cap = entry.min_capability(subtask.task_types) if entry else 0.0
            return RoutingDecision(
                subtask_id=subtask.id,
                assigned_model=subtask.forced_model,
                assigned_model_display=entry.display_name if entry else subtask.forced_model,
                capability_score=cap,
                estimated_cost_usd=cost,
                ranked_alternatives=[],
                override_reason="forced_model",
                verify=verify,
            )

        # 3. Filter available models
        candidates = self._registry.models_for_providers(self._config.enabled_providers)
        qualified = self._filter_and_score(
            candidates, subtask, threshold
        )

        # 4. If none qualify, try relaxing threshold once
        if not qualified:
            relaxed = max(0.0, threshold - 0.05)
            qualified = self._filter_and_score(candidates, subtask, relaxed)
            if not qualified:
                raise RoutingError(
                    f"No model qualifies for sub-task '{subtask.id}' "
                    f"at threshold {threshold:.2f} (tried {relaxed:.2f} too). "
                    f"Lower quality_threshold or add provider API keys."
                )

        # 5. Sort: cost ASC, then capability_score DESC on ties
        qualified.sort(key=lambda x: (x[1], -x[2]))

        best_model, best_cost, best_cap = qualified[0]
        alternatives = [m.id for m, _, _ in qualified[1:4]]

        return RoutingDecision(
            subtask_id=subtask.id,
            assigned_model=best_model.id,
            assigned_model_display=best_model.display_name,
            capability_score=best_cap,
            estimated_cost_usd=best_cost,
            ranked_alternatives=alternatives,
            verify=verify,
        )

    def _filter_and_score(
        self,
        candidates: list[ModelEntry],
        subtask: SubTask,
        threshold: float,
    ) -> list[tuple[ModelEntry, float, float]]:
        """Return (model, estimated_cost, capability_score) for qualifying models."""
        results = []
        needed_tokens = subtask.estimated_input_tokens + subtask.estimated_output_tokens
        for model in candidates:
            cap = model.min_capability(subtask.task_types)
            if cap < threshold:
                continue
            if model.context_window < needed_tokens:
                continue
            cost = model.estimated_cost(
                subtask.estimated_input_tokens, subtask.estimated_output_tokens
            )
            results.append((model, cost, cap))
        return results

    # ── Parallel-group detection (topological sort) ────────────────────────────

    def _compute_parallel_groups(self, subtasks: list[SubTask]) -> list[list[str]]:
        """
        Kahn's algorithm on the depends_on DAG.
        Returns waves: each inner list is a set of task IDs with no inter-dependencies,
        safe to run concurrently.
        """
        id_set = {s.id for s in subtasks}
        in_degree: dict[str, int] = defaultdict(int)
        dependents: dict[str, list[str]] = defaultdict(list)

        for subtask in subtasks:
            in_degree.setdefault(subtask.id, 0)
            for dep in subtask.depends_on:
                if dep in id_set:
                    in_degree[subtask.id] += 1
                    dependents[dep].append(subtask.id)

        queue: deque[str] = deque(
            sid for sid in id_set if in_degree[sid] == 0
        )
        waves: list[list[str]] = []

        while queue:
            wave = list(queue)
            queue.clear()
            waves.append(wave)
            for sid in wave:
                for dep_sid in dependents[sid]:
                    in_degree[dep_sid] -= 1
                    if in_degree[dep_sid] == 0:
                        queue.append(dep_sid)

        # Detect circular dependency
        visited = {sid for wave in waves for sid in wave}
        unvisited = id_set - visited
        if unvisited:
            raise RoutingError(
                f"Circular dependency detected among sub-tasks: {unvisited}"
            )

        return waves

    def _annotate_groups(
        self,
        decisions: list[RoutingDecision],
        parallel_groups: list[list[str]],
    ) -> None:
        id_to_group: dict[str, int] = {}
        for idx, wave in enumerate(parallel_groups):
            for sid in wave:
                id_to_group[sid] = idx
        for decision in decisions:
            decision.parallel_group = id_to_group.get(decision.subtask_id)

    # ── Cost comparison ────────────────────────────────────────────────────────

    def _build_comparison(self, subtasks: list[SubTask]) -> dict[str, float]:
        """Cost if every sub-task were run through each flagship model."""
        comparison: dict[str, float] = {}
        for model_id in _COMPARISON_MODELS:
            entry = self._registry.get(model_id)
            if not entry:
                continue
            total = sum(
                entry.estimated_cost(s.estimated_input_tokens, s.estimated_output_tokens)
                for s in subtasks
            )
            comparison[entry.display_name] = total
        return comparison
