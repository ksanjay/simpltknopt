# SimplTknOpt

**Token-optimized model router** — automatically routes each sub-task of a complex pipeline to the cheapest model capable of satisfying it, across Claude, OpenAI, Gemini, and DeepSeek.

## Quickstart

```bash
pip install simpltknopt
stko init          # save your preferred quality threshold
stko doctor        # verify API keys and registry
```

```python
from simpltknopt import SimplTknOpt

stko = SimplTknOpt()
result = stko.run("Build a REST API for user management")
print(f"Total cost: ${result.total_actual_cost_usd:.4f}")
```

Before executing, you see a routing plan:

```
╔═════════════════════════════════════════════════════════════════════╗
║  SimplTknOpt Routing Plan — "Build a REST API for user management"  ║
╠═══╦══════════════════════════╦═══════════════════╦═════════════╦═══╣
║ # ║ Sub-task                 ║ Model             ║ Tokens      ║ $ ║
╠═══╬══════════════════════════╬═══════════════════╬═════════════╬═══╣
║   ║ ── Wave 1 ─────────────────────────────────────────────────── ║
║ 1 ║ Plan API routes          ║ Gemini 2.0 Flash  ║ 1.2k / 0.8k ║   ║
║   ║ ── Wave 2 (parallel v2) ───────────────────────────────────── ║
║ 2 ║ Write data models        ║ DeepSeek-V3       ║ 2.0k / 3.0k ║   ║
║ 3 ║ Write endpoint handlers  ║ DeepSeek-V3       ║ 3.5k / 5.0k ║   ║
╚═════════════════════════════════════════════════════════════════════╝
  Optimized: $0.026  vs. all-Opus: $0.84  (32×)
```

## Key features

- **Pre-execution plan** — see exactly which model handles each sub-task, with cost estimates and savings vs. flagship models
- **Quality threshold** — models must score above your configured threshold (default 0.75) for the task type; raise it per sub-task for high-stakes work
- **Opt-in verification** — set `verify=True` on any sub-task to have a judge model score the output; auto-escalates to the next-best model on failure
- **Parallel-group detection** — the router identifies which sub-tasks have no mutual dependencies and displays them as parallel waves (ready for v2 async execution)
- **Developer override** — at the plan preview prompt, choose `[O]` to reassign any sub-task to a different model
- **URL-fetched registry** — model pricing and capabilities are always current; locally cached with ETag

## Installation

```bash
pip install simpltknopt
```

Requires Python 3.11+. Set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
export DEEPSEEK_API_KEY=sk-...
```

## CLI

```bash
stko init                                     # first-time setup
stko plan "Summarize 10 documents"            # preview routing plan
stko run  "Summarize 10 documents"            # plan + execute
stko run  "..." --threshold 0.85 --verify     # stricter threshold + verification
stko run  "..." --no-interactive              # CI/CD mode
stko models list                              # show all models + pricing
stko models list --task-type code_generation  # sorted by capability
stko cost-estimate --input-tokens 5000 --output-tokens 2000
stko config show                              # effective configuration
stko config set quality_threshold 0.85        # update persisted preference
stko registry refresh                         # force registry re-fetch
stko doctor                                   # check API keys + registry
```

## Python SDK

```python
from simpltknopt import SimplTknOpt, SubTask, TaskType

stko = SimplTknOpt()

# Manual sub-tasks with selective verification
subtasks = [
    SubTask(
        description="Write OpenAPI spec for /users CRUD",
        task_types=[TaskType.CODE_GENERATION],
        estimated_input_tokens=1000,
        estimated_output_tokens=2000,
    ),
    SubTask(
        description="Security review of the spec",
        task_types=[TaskType.CODE_REVIEW],
        depends_on=["write-openapi-spec-for-users"],
        quality_threshold=0.85,
        verify=True,  # run the judge on this high-stakes output
    ),
]

plan = stko.plan_from_subtasks(subtasks, task_summary="OpenAPI + security review")
result = stko.execute(plan, interactive=False)
print(f"Saved ${result.total_saved_vs_flagship:.4f} vs flagship")
```

## Configuration

Copy `stko.yaml.example` → `stko.yaml` in your project root:

```yaml
defaults:
  quality_threshold: 0.75
  verify: false
  max_escalations: 2
  interactive: true

registry_url: https://registry.simpltknopt.dev/models.yaml

enabled_providers: [anthropic, openai, google, deepseek]
```

User-level defaults (threshold, etc.) are persisted to `~/.stko/preferences.yaml` via `stko init` or `stko config set`.

## License

MIT
