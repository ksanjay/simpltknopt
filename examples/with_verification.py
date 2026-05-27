"""Selective verification: only opt-in high-stakes sub-tasks."""
from simpltknopt import SimplTknOpt, SubTask, TaskType

stko = SimplTknOpt()

subtasks = [
    # Low-stakes: no verification needed
    SubTask(
        description="Draft a product description for a noise-cancelling headset",
        task_types=[TaskType.CREATIVE_WRITING],
        estimated_input_tokens=300,
        estimated_output_tokens=400,
        verify=False,
    ),
    # High-stakes: verify and escalate if quality is low
    SubTask(
        description="Translate the product description to French, preserving tone",
        task_types=[TaskType.TRANSLATION],
        depends_on=["draft-a-product-description"],
        estimated_input_tokens=500,
        estimated_output_tokens=500,
        quality_threshold=0.85,
        verify=True,
    ),
]

plan = stko.plan_from_subtasks(subtasks, task_summary="Product copy + French translation")
result = stko.execute(plan, interactive=True)

for r in result.subtask_results:
    status = "✓" if r.quality_passed else "✗"
    verified = f" (verified, score={r.quality_score:.2f})" if r.verified else ""
    print(f"\n[{status}] {r.subtask_id}{verified}")
    print(r.output)
