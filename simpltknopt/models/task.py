"""Core data models for SimplTknOpt task routing pipeline."""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskType(str, Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    DATA_EXTRACTION = "data_extraction"
    CREATIVE_WRITING = "creative_writing"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    QA = "qa"
    INSTRUCTION_FOLLOWING = "instruction_following"
    TOOL_USE = "tool_use"


def _slugify(text: str) -> str:
    """Convert a description string into a URL-safe slug."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_]+", "-", slug).strip("-")
    return slug[:48] or "subtask"


class SubTask(BaseModel):
    """A single unit of work to be routed to a model."""

    id: str = Field(default="")
    description: str
    task_types: list[TaskType]
    estimated_input_tokens: int = Field(default=1000, ge=1)
    estimated_output_tokens: int = Field(default=500, ge=1)
    depends_on: list[str] = Field(default_factory=list)
    quality_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    verify: bool = False
    forced_model: Optional[str] = None
    context_passthrough: bool = True

    @model_validator(mode="after")
    def _set_id(self) -> "SubTask":
        if not self.id:
            self.id = _slugify(self.description)
        return self

    @field_validator("task_types")
    @classmethod
    def _non_empty(cls, v: list[TaskType]) -> list[TaskType]:
        if not v:
            raise ValueError("task_types must contain at least one TaskType")
        return v


class RoutingDecision(BaseModel):
    """The model assignment for one sub-task, produced by the Router."""

    subtask_id: str
    assigned_model: str
    assigned_model_display: str
    capability_score: float
    estimated_cost_usd: float
    ranked_alternatives: list[str] = Field(default_factory=list)
    override_reason: Optional[str] = None
    parallel_group: Optional[int] = None
    verify: bool = False


class RoutingPlan(BaseModel):
    """Full routing plan for a task: all sub-tasks, model assignments, cost estimates."""

    task_summary: str
    subtasks: list[SubTask]
    decisions: list[RoutingDecision]
    total_estimated_cost_usd: float
    comparison: dict[str, float] = Field(default_factory=dict)
    parallel_groups: list[list[str]] = Field(default_factory=list)
    quality_threshold: float = 0.75
    verify_enabled: bool = False

    def decision_for(self, subtask_id: str) -> Optional[RoutingDecision]:
        for d in self.decisions:
            if d.subtask_id == subtask_id:
                return d
        return None

    def subtask_for(self, subtask_id: str) -> Optional[SubTask]:
        for s in self.subtasks:
            if s.id == subtask_id:
                return s
        return None


class SubTaskResult(BaseModel):
    """Outcome of executing a single sub-task."""

    subtask_id: str
    model_used: str
    output: str
    quality_score: float = -1.0
    quality_passed: bool = True
    escalation_count: int = 0
    actual_cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    verified: bool = False


class ExecutionResult(BaseModel):
    """Final result of running a full RoutingPlan."""

    task_summary: str
    subtask_results: list[SubTaskResult]
    total_actual_cost_usd: float
    total_saved_vs_flagship: float
    routing_plan: RoutingPlan

    @property
    def combined_output(self) -> str:
        """Concatenate all sub-task outputs in order."""
        return "\n\n---\n\n".join(
            f"## {r.subtask_id}\n{r.output}" for r in self.subtask_results
        )
