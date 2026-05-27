"""Custom verification rubric: append domain-specific criteria to the judge prompt."""
from simpltknopt import SimplTknOpt, SubTask, TaskType
from simpltknopt.verifier import Verifier
from simpltknopt.registry import get_registry
from simpltknopt.config import get_config

cfg = get_config()
reg = get_registry(config=cfg)

# Domain-specific rubric appended to the default judge prompt
SECURITY_RUBRIC = """\
Additional security-focused criteria:
- Must explicitly mention authentication/authorization implications
- Must flag any SQL injection, XSS, or SSRF risks if present
- Must note OWASP Top 10 relevance where applicable
"""

verifier = Verifier(registry=reg, config=cfg, custom_rubric=SECURITY_RUBRIC)
stko = SimplTknOpt()

# Override the executor's verifier
from simpltknopt.executor import Executor
executor = Executor(registry=reg, config=cfg, verifier=verifier)

subtasks = [
    SubTask(
        description="Review the following Python Flask route handler for security vulnerabilities",
        task_types=[TaskType.CODE_REVIEW],
        estimated_input_tokens=1500,
        estimated_output_tokens=1000,
        quality_threshold=0.80,
        verify=True,
    ),
]

plan = stko.plan_from_subtasks(subtasks, task_summary="Security code review")
# Execute directly with the custom executor
result = executor.run(plan)
print(result.combined_output)
