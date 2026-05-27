"""Manually define sub-tasks — bypass decomposition entirely."""
from simpltknopt import SimplTknOpt, SubTask, TaskType

stko = SimplTknOpt()

subtasks = [
    SubTask(
        description="Write an OpenAPI 3.1 spec for a /users CRUD REST API",
        task_types=[TaskType.CODE_GENERATION],
        estimated_input_tokens=800,
        estimated_output_tokens=2000,
    ),
    SubTask(
        id="review-openapi-spec",
        description="Review the OpenAPI spec for security issues and missing edge cases",
        task_types=[TaskType.CODE_REVIEW],
        depends_on=["write-an-openapi-31-spec-for-a-users-crud"],
        estimated_input_tokens=2500,
        estimated_output_tokens=800,
        quality_threshold=0.85,
        verify=True,  # this one is high-stakes, run the judge
    ),
]

plan = stko.plan_from_subtasks(subtasks, task_summary="OpenAPI spec + security review")
result = stko.execute(plan, interactive=False)

for r in result.subtask_results:
    print(f"\n=== {r.subtask_id} ({r.model_used}) ===")
    print(r.output[:500])
