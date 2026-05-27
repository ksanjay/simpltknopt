# SimplTknOpt Developer Guide

**Token-optimized model routing across Claude, OpenAI, Gemini, and DeepSeek**

This guide walks you through getting set up and using SimplTknOpt in your enterprise environment. No advanced Python knowledge required — if you can run a `pip install` and set an environment variable, you're ready to go.

---

## What does SimplTknOpt actually do?

Every time you call a large language model, you pay per token. Flagship models like Claude Opus or GPT-4o are powerful but expensive. Many sub-tasks in a real workflow (summarizing a document, translating a string, classifying a ticket) don't need flagship power — a faster, cheaper model handles them just as well.

SimplTknOpt automatically:

1. **Breaks your task into sub-tasks** using a lightweight decomposition model
2. **Scores every available model** against each sub-task based on capability and cost
3. **Shows you a routing plan** — which model handles which sub-task, estimated tokens, and projected cost
4. **Executes the plan** and hands you back the combined output

You see the cost breakdown before anything runs. You can override any assignment. You stay in control.

---

## Installation

```bash
pip install simpltknopt
```

Requires Python 3.10 or higher.

### First-time setup

Run the setup wizard once to save your preferences:

```bash
stko init
```

This saves your preferred quality threshold to `~/.stko/preferences.yaml` so you don't have to pass it on every run.

### Verify everything is working

```bash
stko doctor
```

This checks your API keys, confirms the model registry loaded, and lists which providers are active.

---

## Setting your API keys

SimplTknOpt reads API keys from environment variables. In your enterprise environment you may already have these set via a secrets manager, `.env` file, or shell profile. If not, set them manually:

```bash
# Only set the keys for providers your enterprise has access to.
# You do not need all four — the router will only use what's available.

export ANTHROPIC_API_KEY="sk-ant-..."     # Claude models
export OPENAI_API_KEY="sk-..."            # GPT-4o, o3-mini
export GOOGLE_API_KEY="..."               # Gemini models
export DEEPSEEK_API_KEY="sk-..."          # DeepSeek-V3, DeepSeek-R1
```

> **Enterprise tip:** If your organization restricts which providers you can call, set `enabled_providers` in `stko.yaml` (see Configuration below). The router will only route to those providers.

---

## Project configuration (optional but recommended)

Copy the example config into your project root:

```bash
cp stko.yaml.example stko.yaml
```

Then edit `stko.yaml` to match your setup:

```yaml
# stko.yaml — commit this file to your repo (no secrets here)

defaults:
  quality_threshold: 0.75   # 0.0–1.0; raise this for high-stakes work
  verify: false             # set true to run LLM quality checks on outputs
  interactive: true         # set false for CI/CD pipelines

# Restrict to providers your company has approved
enabled_providers:
  - anthropic
  - openai
```

The config file is optional — sensible defaults are used if it's absent.

---

## Concepts in 60 seconds

| Term | What it means |
|---|---|
| **SubTask** | One discrete unit of work (e.g. "Write a SQL query", "Summarize this doc") |
| **TaskType** | The category of work: `code_generation`, `summarization`, `qa`, `reasoning`, etc. |
| **Quality threshold** | Minimum capability score (0–1) a model must have for a TaskType before it can be assigned |
| **Routing plan** | The full assignment table shown before execution |
| **Verify** | Opt-in: run a judge model to score the output and escalate to a better model if it fails |
| **Parallel group** | Sub-tasks with no dependencies on each other — shown as a "wave" in the plan table |

---

## Example 1 — One line of code (auto-decompose)

The simplest possible usage. Give SimplTknOpt a plain-English description of your task and let it handle everything.

```python
from simpltknopt import SimplTknOpt

stko = SimplTknOpt()
result = stko.run("Build a REST API for user authentication with JWT tokens")

# The combined output from all sub-tasks
print(result.combined_output)

# How much it cost
print(f"Total cost:  ${result.total_actual_cost_usd:.4f}")
print(f"Savings vs flagship: ${result.total_saved_vs_flagship:.4f}")
```

**What happens when you run this:**

1. The terminal shows a routing plan table — you'll see each sub-task, which model was chosen, estimated tokens, and cost.
2. You're prompted: `[Y]es / [N]o / [O]verride`. Type `Y` to proceed, `O` to reassign any sub-task to a different model.
3. SimplTknOpt executes each sub-task sequentially and returns the combined result.

**Sample output before execution:**

```
╔══════════════════════════════════════════════════════════════════╗
║  SimplTknOpt — "Build a REST API for user authentication..."      ║
╠═══╦══════════════════════════╦═══════════════════╦═════════╦════╣
║ # ║ Sub-task                 ║ Model             ║ Tokens  ║  $ ║
╠═══╬══════════════════════════╬═══════════════════╬═════════╬════╣
║   ║ ── Wave 1 ──────────────────────────────────────────────── ║
║ 1 ║ Plan API routes          ║ Gemini 2.0 Flash  ║ 1.2k    ║    ║
║   ║ ── Wave 2 (parallel) ───────────────────────────────────── ║
║ 2 ║ Write JWT middleware     ║ DeepSeek-V3       ║ 2.5k    ║    ║
║ 3 ║ Write endpoint handlers  ║ DeepSeek-V3       ║ 3.0k    ║    ║
║   ║ ── Wave 3 ──────────────────────────────────────────────── ║
║ 4 ║ Write unit tests         ║ GPT-4o mini       ║ 2.0k    ║    ║
╚══════════════════════════════════════════════════════════════════╝
  Optimized: $0.031  vs. all-Opus: $0.97  (31×)
```

**When to use this:** exploratory work, prototyping, tasks where you trust the decomposer to figure out the sub-tasks.

---

## Example 2 — Manual sub-tasks with selective verification

When you know exactly what needs to happen, define the sub-tasks yourself. This gives you precise control over which task types each step is scored against, and lets you turn on verification only for the steps that matter.

This pattern is ideal for high-stakes enterprise workflows — for example, generating and then security-reviewing a configuration file.

```python
from simpltknopt import SimplTknOpt, SubTask, TaskType

stko = SimplTknOpt()

subtasks = [
    # Step 1: Generate the Terraform config
    SubTask(
        description="Write Terraform config for an AWS ECS cluster with autoscaling",
        task_types=[TaskType.CODE_GENERATION],
        estimated_input_tokens=800,
        estimated_output_tokens=2000,
    ),

    # Step 2: Security review — runs after step 1, and uses verification
    # so a judge model scores the output before accepting it
    SubTask(
        description="Security review of the Terraform config — check for overly permissive IAM roles, open security groups, and unencrypted storage",
        task_types=[TaskType.CODE_REVIEW],
        depends_on=["write-terraform-config-for-an-aws-ecs-cluster-with-autoscaling"],
        quality_threshold=0.85,   # Stricter threshold for security work
        verify=True,              # Run the judge — escalate if it fails
    ),

    # Step 3: Write a summary for the team, runs after review
    SubTask(
        description="Write a one-page summary of the infrastructure design for the engineering team",
        task_types=[TaskType.SUMMARIZATION],
        depends_on=["security-review-of-the-terraform-config"],
        estimated_input_tokens=500,
        estimated_output_tokens=400,
    ),
]

plan = stko.plan_from_subtasks(
    subtasks,
    task_summary="Terraform ECS cluster + security review"
)

result = stko.execute(plan)

# Access individual sub-task outputs
for sub_result in result.subtask_results:
    print(f"\n=== {sub_result.subtask_id} ===")
    print(f"Model used: {sub_result.model_used}")
    print(f"Cost: ${sub_result.actual_cost_usd:.5f}")
    if sub_result.verified:
        print(f"Quality score: {sub_result.quality_score:.2f} ✓")
    print(sub_result.output[:500])  # first 500 chars
```

**Key things to notice:**

- `depends_on` takes the auto-generated ID of a previous sub-task. The ID is a URL-safe slug of the description (lowercase, hyphens). You can also set `id="my-custom-id"` explicitly to make this easier.
- `quality_threshold=0.85` means the router will only consider models that score 0.85 or above for `code_review`. This pushes the security review toward a stronger model even if it costs more.
- `verify=True` means after the model responds, a judge model reads the output and scores it. If the score is below threshold, the executor automatically retries with the next-best model (up to `max_escalations` times).

**When to use this:** automated pipelines, compliance workflows, any task where you need an audit trail of which model ran what.

---

## Example 3 — CI/CD pipeline (non-interactive, strict threshold)

In a CI/CD context you don't want a human approval prompt blocking the pipeline. Set `interactive=False` and the plan executes immediately.

This example also shows how to catch quality failures gracefully and report them in your pipeline output.

```python
import sys
from simpltknopt import SimplTknOpt, SubTask, TaskType

# Non-interactive: no terminal prompts, runs straight through
stko = SimplTknOpt()

subtasks = [
    SubTask(
        description="Summarize the pull request diff into a one-paragraph changelog entry",
        task_types=[TaskType.SUMMARIZATION],
        estimated_input_tokens=3000,
        estimated_output_tokens=200,
    ),
    SubTask(
        description="Classify the PR as: feature, bugfix, refactor, docs, or chore",
        task_types=[TaskType.CLASSIFICATION],
        depends_on=["summarize-the-pull-request-diff-into-a-one-paragraph-changelog-entry"],
        estimated_input_tokens=300,
        estimated_output_tokens=20,
    ),
    SubTask(
        description="Check the PR for any security-sensitive changes (secrets, auth logic, SQL queries)",
        task_types=[TaskType.CODE_REVIEW],
        quality_threshold=0.88,   # High bar — only strong models for security
        verify=True,
        estimated_input_tokens=3000,
        estimated_output_tokens=500,
    ),
]

plan = stko.plan_from_subtasks(
    subtasks,
    task_summary="PR automation: changelog + classification + security scan",
    threshold=0.80,    # Minimum capability for non-security tasks
    show=True,         # Still prints the routing table to CI logs
)

# Execute without prompting
result = stko.execute(plan, interactive=False)

# Report results
print(f"\n✅ Cost: ${result.total_actual_cost_usd:.4f}")
print(f"💰 Saved vs flagship: ${result.total_saved_vs_flagship:.4f}\n")

# Check for any quality failures
failures = [r for r in result.subtask_results if not r.quality_passed]
if failures:
    for f in failures:
        print(f"⚠️  Quality check failed: {f.subtask_id} (score: {f.quality_score:.2f})")
    sys.exit(1)  # Fail the CI step

# Print outputs for downstream pipeline steps
for sub_result in result.subtask_results:
    print(f"\n--- {sub_result.subtask_id} ---")
    print(sub_result.output)
```

**Running this from the command line instead:**

```bash
stko run "Summarize and classify this PR" \
  --threshold 0.80 \
  --verify \
  --no-interactive
```

**When to use this:** GitHub Actions, GitLab CI, any automated workflow where you want cost-controlled LLM calls with no human in the loop.

---

## CLI quick reference

```bash
stko init                                  # First-time setup, saves preferences
stko doctor                                # Check API keys and registry health
stko plan "Your task description"          # Preview routing plan (no execution)
stko run  "Your task description"          # Plan + execute interactively
stko run  "..." --threshold 0.85           # Raise quality bar for this run
stko run  "..." --verify                   # Enable output verification
stko run  "..." --no-interactive           # Skip the approval prompt (CI mode)
stko models list                           # Show all models with pricing
stko models list --task-type code_review   # Filter by task type, sorted by score
stko cost-estimate --input-tokens 5000 --output-tokens 2000  # Quick cost calc
stko config show                           # Print effective configuration
stko config set quality_threshold 0.85     # Save a new default
stko registry refresh                      # Force re-fetch of model pricing
```

---

## Quality threshold guide

The threshold is the minimum capability score (0.0 to 1.0) a model must have for a given task type before it can be assigned. Start here and adjust:

| Threshold | When to use |
|---|---|
| **0.70** | Drafts, internal tools, tasks where speed matters more than polish |
| **0.75** | Default — good balance for most enterprise workflows |
| **0.80** | Customer-facing content, business logic, anything that gets reviewed |
| **0.85** | Security reviews, compliance checks, high-stakes decisions |
| **0.90+** | Research, legal, medical — reserve for critical outputs only |

You can mix thresholds within a single run by setting `quality_threshold` on individual `SubTask` objects, as shown in Example 2.

---

## Troubleshooting

**"No models available for sub-task"**
The threshold is too high for the providers you have enabled. Try lowering `quality_threshold`, or check that the relevant provider is in `enabled_providers` and its API key is set.

**"stko doctor shows a provider as missing"**
The API key environment variable is not set in the current shell session. Re-export it or add it to your shell profile.

**Costs look higher than expected**
Check `stko models list` — the registry might be stale. Run `stko registry refresh` to pull the latest pricing.

**The decomposer is splitting the task in an unexpected way**
Switch to manual sub-tasks (Example 2) for full control. The auto-decomposer is a convenience, not a requirement.

---

## Getting help

- Run `stko --help` or `stko <command> --help` for CLI documentation
- See `stko.yaml.example` in your project root for all configuration options
- The model registry is at `registry/models.yaml` — you can inspect capability scores there
