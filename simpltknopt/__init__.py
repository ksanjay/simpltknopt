"""SimplTknOpt — token-optimized model router."""
from __future__ import annotations

from typing import Optional

from .config import Config, get_config
from .decomposer import Decomposer
from .executor import Executor
from .models.task import (
    ExecutionResult,
    RoutingPlan,
    SubTask,
    SubTaskResult,
    TaskType,
)
from .planner import render_plan, prompt_proceed
from .registry import ModelRegistry, get_registry
from .router import Router
from .verifier import Verifier

__all__ = [
    "SimplTknOpt",
    "SubTask",
    "TaskType",
    "RoutingPlan",
    "ExecutionResult",
    "SubTaskResult",
    "Config",
    "Router",
    "Decomposer",
    "Executor",
    "Verifier",
    "ModelRegistry",
]

__version__ = "0.1.0"


class SimplTknOpt:
    """High-level facade: decompose → route → preview → execute."""

    def __init__(
        self,
        config: Optional[Config] = None,
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._config = config or get_config()
        self._registry = registry or get_registry(config=self._config)
        self._router = Router(registry=self._registry, config=self._config)
        self._decomposer = Decomposer(config=self._config)
        self._verifier = Verifier(registry=self._registry, config=self._config)
        self._executor = Executor(
            registry=self._registry,
            config=self._config,
            verifier=self._verifier,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        threshold: Optional[float] = None,
        verify: Optional[bool] = None,
        interactive: Optional[bool] = None,
    ) -> ExecutionResult:
        """One-shot: decompose → route → (optionally preview) → execute."""
        plan = self.plan(task, threshold=threshold, verify=verify)
        do_interactive = interactive if interactive is not None else self._config.interactive
        if do_interactive:
            should_run, plan = prompt_proceed(plan)
            if not should_run:
                raise RuntimeError("Execution cancelled by user.")
        return self._executor.run(plan)

    def plan(
        self,
        task: str,
        threshold: Optional[float] = None,
        verify: Optional[bool] = None,
        show: bool = True,
    ) -> RoutingPlan:
        """Decompose a task and build (and optionally display) a RoutingPlan."""
        subtasks = self._decomposer.decompose(task)
        routing_plan = self._router.plan(
            subtasks,
            task_summary=task,
            run_threshold=threshold,
            run_verify=verify,
        )
        if show:
            render_plan(routing_plan)
        return routing_plan

    def plan_from_subtasks(
        self,
        subtasks: list[SubTask],
        task_summary: str = "",
        threshold: Optional[float] = None,
        verify: Optional[bool] = None,
        show: bool = True,
    ) -> RoutingPlan:
        """Build a RoutingPlan from a manually supplied SubTask list."""
        routing_plan = self._router.plan(
            subtasks,
            task_summary=task_summary,
            run_threshold=threshold,
            run_verify=verify,
        )
        if show:
            render_plan(routing_plan)
        return routing_plan

    def execute(
        self,
        plan: RoutingPlan,
        interactive: Optional[bool] = None,
    ) -> ExecutionResult:
        """Execute a RoutingPlan (with optional interactive approval)."""
        do_interactive = interactive if interactive is not None else self._config.interactive
        if do_interactive:
            should_run, plan = prompt_proceed(plan)
            if not should_run:
                raise RuntimeError("Execution cancelled by user.")
        return self._executor.run(plan)
