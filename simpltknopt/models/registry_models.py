"""Data models for the model capability registry."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .task import TaskType


class PricingInfo(BaseModel):
    input_per_mtok: float = Field(..., description="USD per million input tokens")
    output_per_mtok: float = Field(..., description="USD per million output tokens")

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000 * self.input_per_mtok) + (
            output_tokens / 1_000_000 * self.output_per_mtok
        )


CapabilityMap = dict[str, float]


class ModelEntry(BaseModel):
    id: str
    display_name: str
    provider: str
    pricing: PricingInfo
    context_window: int
    capabilities: CapabilityMap = Field(default_factory=dict)
    supports_tool_use: bool = True
    supports_vision: bool = False
    notes: Optional[str] = None

    def capability_score(self, task_type: TaskType) -> float:
        return self.capabilities.get(task_type.value, 0.0)

    def min_capability(self, task_types: list[TaskType]) -> float:
        """Minimum capability across all requested task types — the model must satisfy ALL."""
        if not task_types:
            return 0.0
        return min(self.capability_score(t) for t in task_types)

    def estimated_cost(self, input_tokens: int, output_tokens: int) -> float:
        return self.pricing.cost(input_tokens, output_tokens)
