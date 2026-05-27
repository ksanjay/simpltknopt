# SimplTknOpt — Token-Optimized Model Router
## Plan & Specification (Pre-Implementation Review Draft)

> **Status:** Approved — ready for implementation.

---

## 1. Problem Statement

Running every sub-task of a complex pipeline through a flagship model (Claude Opus, GPT-4o, Gemini 1.5 Pro) is the path of least resistance — and the most expensive one. Many sub-tasks only require a fraction of that capability. The gap between "what a task needs" and "what the model costs" is where money burns.

This library gives developers a routing layer that:

1. **Decomposes** a high-level task into typed sub-tasks
2. **Previews** exactly which model will handle each sub-task, with a cost comparison table, before any API call is made
3. **Routes** each sub-task to the cheapest model capable of satisfying it above a configurable quality threshold
4. **Verifies** the output of each sub-task before passing it downstream
5. **Escalates** automatically to the next-cheapest qualifying model if quality fails

---

## 2. Scope

### In scope
- Python library + CLI tool
- Providers: Anthropic (Claude), OpenAI (GPT/o-series), Google (Gemini), DeepSeek
- Routing plan preview with cost comparison (human-readable + machine-readable)
- Quality verification with configurable rubrics
- Automatic escalation on quality failure
- Per-task override / developer annotation
- Static model capability registry (YAML) that developers can extend or override

### Out of scope (v1)
- Streaming responses
- Async parallel execution (v2)
- Fine-tuned model support
- Automatic registry calibration via benchmarks (v2)
- Web UI

---

## 3. Architecture Overview

> See the interactive architecture diagram in the project's Cowork session (click any node to explore that component).

The pipeline runs top-to-bottom through seven stages. The **Model Registry** feeds into the Routing Engine as a side input. The **Plan Preview** stage is the one explicit human checkpoint — the developer sees the full cost table and can override any routing decision before execution begins. The dashed **fail → retry** loop on the right shows that when the Verifier rejects a sub-task output, the Executor re-runs it with the next-cheapest qualified model (up to `max_escalations`).

| Stage | Color | Role |
|---|---|---|
| Decomposer, Routing Engine, Executor, Verifier | Purple | Core pipeline |
| Model Registry | Teal | YAML data store |
| Plan Preview | Coral | Developer interaction point |
| Developer Input, Execution Result | Gray | I/O boundaries |

---

## 4. Open Source Dependencies

All dependencies are MIT or Apache 2.0 licensed.

| Library      | Role                                          | License    |
|--------------|-----------------------------------------------|------------|
| **LiteLLM**  | Unified API calls to all providers            | MIT        |
| **Rich**     | Terminal routing plan table + progress display | MIT        |
| **Pydantic** | Data models, validation, serialization        | MIT        |
| **Typer**    | CLI framework                                 | MIT        |
| **PyYAML**   | Model registry config file parsing            | MIT        |
| **httpx**    | Async-capable HTTP client for registry fetch  | BSD        |
| **tenacity** | Retry logic with exponential backoff          | Apache 2.0 |
| **tiktoken** | Token counting for OpenAI-compatible models   | MIT        |
| **pytest**   | Testing                                       | MIT        |

**Why LiteLLM as the backbone:** It provides a single `completion()` call interface across all four providers, handles auth, and exposes cost tracking via `litellm.completion_cost()`. It is well-maintained, has 15k+ GitHub stars, and avoids vendor lock-in.

---

## 5. Model Capability Registry

### 5.1 Design

The registry is a YAML document that is the single source of truth for:
- Model IDs (as used by LiteLLM)
- Input and output token pricing (per million tokens)
- Context window size
- Capability scores (0.0–1.0) per task type
- Known hard limits (e.g., no function calling, no vision)

**Fetch strategy:** On startup, `registry.py` fetches the latest registry from a hosted URL (default: `https://registry.simpltknopt.dev/models.yaml`). The fetched document is cached locally at `~/.stko/registry_cache.yaml` with an ETag/Last-Modified header so subsequent startups only re-download when the registry has actually changed. If the network is unavailable, the last cached version is used with a warning. The cache TTL is 24 hours; `stko registry refresh` forces an immediate re-fetch.

Developers can:
- Accept the hosted registry as-is (recommended for up-to-date pricing)
- Point to a self-hosted URL via `registry_url` in `stko.yaml` (air-gapped environments)
- Extend via a local `stko.yaml` overlay (add custom models, adjust scores)
- Override individual scores for their domain

### 5.2 Task Type Taxonomy

```
code_generation        — Writing new code from a spec
code_review            — Reading code, identifying issues, suggesting fixes
reasoning              — Multi-step logical, mathematical, or causal reasoning
summarization          — Condensing long content to key points
data_extraction        — Structured extraction from unstructured text
creative_writing       — Narrative, marketing copy, tone-sensitive text
classification         — Categorizing inputs into predefined buckets
translation            — Language-to-language translation
qa                     — Question answering over provided documents
instruction_following  — Complex multi-constraint instruction adherence
tool_use               — Function/tool calling with structured output
```

### 5.3 Registry Schema (abbreviated)

```yaml
# registry/models.yaml
models:
  - id: claude-haiku-4-5-20251001
    display_name: "Claude Haiku 4.5"
    provider: anthropic
    pricing:
      input_per_mtok: 0.80
      output_per_mtok: 4.00
    context_window: 200000
    capabilities:
      code_generation: 0.72
      code_review: 0.70
      reasoning: 0.65
      summarization: 0.88
      data_extraction: 0.85
      creative_writing: 0.75
      classification: 0.92
      translation: 0.87
      qa: 0.82
      instruction_following: 0.75
      tool_use: 0.78

  - id: claude-sonnet-4-6
    display_name: "Claude Sonnet 4.6"
    provider: anthropic
    pricing:
      input_per_mtok: 3.00
      output_per_mtok: 15.00
    context_window: 200000
    capabilities:
      code_generation: 0.90
      code_review: 0.90
      reasoning: 0.87
      summarization: 0.92
      data_extraction: 0.93
      creative_writing: 0.89
      classification: 0.95
      translation: 0.91
      qa: 0.92
      instruction_following: 0.93
      tool_use: 0.92

  - id: claude-opus-4-6
    display_name: "Claude Opus 4.6"
    provider: anthropic
    pricing:
      input_per_mtok: 15.00
      output_per_mtok: 75.00
    context_window: 200000
    capabilities:
      code_generation: 0.97
      code_review: 0.97
      reasoning: 0.97
      summarization: 0.96
      data_extraction: 0.96
      creative_writing: 0.97
      classification: 0.97
      translation: 0.95
      qa: 0.96
      instruction_following: 0.98
      tool_use: 0.96

  - id: gpt-4o-mini
    display_name: "GPT-4o mini"
    provider: openai
    pricing:
      input_per_mtok: 0.15
      output_per_mtok: 0.60
    context_window: 128000
    capabilities:
      code_generation: 0.78
      code_review: 0.75
      reasoning: 0.70
      summarization: 0.85
      data_extraction: 0.82
      creative_writing: 0.76
      classification: 0.88
      translation: 0.87
      qa: 0.83
      instruction_following: 0.78
      tool_use: 0.82

  - id: gpt-4o
    display_name: "GPT-4o"
    provider: openai
    pricing:
      input_per_mtok: 2.50
      output_per_mtok: 10.00
    context_window: 128000
    capabilities:
      code_generation: 0.93
      code_review: 0.92
      reasoning: 0.90
      summarization: 0.93
      data_extraction: 0.94
      creative_writing: 0.91
      classification: 0.95
      translation: 0.93
      qa: 0.93
      instruction_following: 0.93
      tool_use: 0.94

  - id: o3-mini
    display_name: "o3-mini"
    provider: openai
    pricing:
      input_per_mtok: 1.10
      output_per_mtok: 4.40
    context_window: 128000
    capabilities:
      code_generation: 0.88
      code_review: 0.86
      reasoning: 0.97
      summarization: 0.78
      data_extraction: 0.80
      creative_writing: 0.68
      classification: 0.82
      translation: 0.80
      qa: 0.85
      instruction_following: 0.83
      tool_use: 0.80

  - id: gemini/gemini-2.0-flash
    display_name: "Gemini 2.0 Flash"
    provider: google
    pricing:
      input_per_mtok: 0.10
      output_per_mtok: 0.40
    context_window: 1000000
    capabilities:
      code_generation: 0.80
      code_review: 0.78
      reasoning: 0.76
      summarization: 0.88
      data_extraction: 0.87
      creative_writing: 0.78
      classification: 0.90
      translation: 0.88
      qa: 0.86
      instruction_following: 0.80
      tool_use: 0.80

  - id: gemini/gemini-1.5-pro
    display_name: "Gemini 1.5 Pro"
    provider: google
    pricing:
      input_per_mtok: 1.25
      output_per_mtok: 5.00
    context_window: 2000000
    capabilities:
      code_generation: 0.89
      code_review: 0.88
      reasoning: 0.88
      summarization: 0.92
      data_extraction: 0.92
      creative_writing: 0.87
      classification: 0.93
      translation: 0.92
      qa: 0.92
      instruction_following: 0.90
      tool_use: 0.88

  - id: deepseek/deepseek-chat
    display_name: "DeepSeek-V3"
    provider: deepseek
    pricing:
      input_per_mtok: 0.27
      output_per_mtok: 1.10
    context_window: 64000
    capabilities:
      code_generation: 0.90
      code_review: 0.88
      reasoning: 0.85
      summarization: 0.84
      data_extraction: 0.83
      creative_writing: 0.77
      classification: 0.85
      translation: 0.82
      qa: 0.84
      instruction_following: 0.85
      tool_use: 0.80

  - id: deepseek/deepseek-reasoner
    display_name: "DeepSeek-R1"
    provider: deepseek
    pricing:
      input_per_mtok: 0.55
      output_per_mtok: 2.19
    context_window: 64000
    capabilities:
      code_generation: 0.90
      code_review: 0.88
      reasoning: 0.97
      summarization: 0.83
      data_extraction: 0.82
      creative_writing: 0.72
      classification: 0.83
      translation: 0.80
      qa: 0.86
      instruction_following: 0.85
      tool_use: 0.79
```

---

## 6. Core Data Models (`simpltknopt/models/`)

```python
# task.py  — all Pydantic v2 models

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

class SubTask(BaseModel):
    id: str                          # auto-generated slug
    description: str                 # what this sub-task does
    task_types: list[TaskType]       # can be multi-typed
    estimated_input_tokens: int      # rough estimate used for cost preview
    estimated_output_tokens: int
    depends_on: list[str] = []       # ids of upstream sub-tasks
    quality_threshold: float = 0.75  # 0.0–1.0; overrides global default
    verify: bool = False             # opt-in: run Verifier after this sub-task
    forced_model: str | None = None  # developer override
    context_passthrough: bool = True # pass prior sub-task outputs as context

class RoutingDecision(BaseModel):
    subtask_id: str
    assigned_model: str
    assigned_model_display: str
    capability_score: float
    estimated_cost_usd: float
    ranked_alternatives: list[str]   # next-cheapest qualified models
    override_reason: str | None = None
    parallel_group: int | None = None  # set by executor; tasks sharing a group can run concurrently

class RoutingPlan(BaseModel):
    task_summary: str
    subtasks: list[SubTask]
    decisions: list[RoutingDecision]
    total_estimated_cost_usd: float
    comparison: dict[str, float]     # {"all-Opus 4.6": 0.84, ...}
    parallel_groups: list[list[str]] # e.g. [["task-1","task-3"], ["task-2","task-4"], ["task-5"]]
                                     # each inner list is a wave of tasks with no mutual dependencies

class SubTaskResult(BaseModel):
    subtask_id: str
    model_used: str
    output: str
    quality_score: float
    quality_passed: bool
    escalation_count: int = 0
    actual_cost_usd: float
    input_tokens: int
    output_tokens: int

class ExecutionResult(BaseModel):
    task_summary: str
    subtask_results: list[SubTaskResult]
    total_actual_cost_usd: float
    total_saved_vs_plan: float       # vs. most expensive single model
    routing_plan: RoutingPlan
```

---

## 7. Component Specifications

### 7.1 Decomposer (`decomposer.py`)

**Responsibility:** Take a free-form task string and return a list of `SubTask` objects.

**Approach:**
- Uses a small, fast model by default (configurable; default: `claude-haiku-4-5` or `gpt-4o-mini`) to analyze the task
- Returns structured JSON parsed into `SubTask` list via Pydantic
- Falls back to a single-sub-task passthrough if decomposition confidence is low
- Developer can bypass and supply their own `SubTask` list directly

**Key behaviors:**
- Identifies dependencies between sub-tasks (task 3 needs output of task 2)
- Tags each sub-task with one or more `TaskType` values
- Produces token estimates (rough order-of-magnitude; used only for cost preview)
- Always shows the decomposition to the developer before routing (no silent decomposition)

**Interface:**
```python
decomposer = Decomposer(config)
subtasks: list[SubTask] = decomposer.decompose("Build a REST API for user management")
# or skip decomposition:
subtasks = [SubTask(description="Write OpenAPI spec", task_types=[TaskType.CODE_GENERATION], ...)]
```

---

### 7.2 Routing Engine (`router.py`)

**Responsibility:** Given sub-tasks, assign the cheapest model that meets the quality threshold.

**Algorithm (per sub-task):**

```
1. Resolve effective threshold:
   subtask.quality_threshold           (if explicitly set on the SubTask)
   else run_threshold                  (if passed to router.plan() for this run)
   else config.defaults.quality_threshold  (from stko.yaml or persisted prefs)
   else 0.75                           (hard-coded fallback)

2. Compute primary_task_type = task_types[0]  (most demanding type)
3. For each model in registry:
   a. Check: capability_score[primary_task_type] >= effective_threshold
   b. Check: context_window >= estimated_input_tokens + estimated_output_tokens
   c. Check: provider API key is configured
   d. Compute: estimated_cost = (input_toks/1M * input_price) + (output_toks/1M * output_price)
4. Sort passing models by (estimated_cost ASC, capability_score DESC)
   — ties on cost broken by higher capability score, not alphabetically
5. Assign models[0] (cheapest qualifying, highest quality among equals)
6. Store models[1:3] as ranked_alternatives for escalation
7. If forced_model is set on SubTask, use it unconditionally
8. If no model qualifies at threshold: warn developer, lower threshold by 0.05 and retry once,
   then raise RoutingError with suggested threshold adjustment
```

**Multi-type task handling:** For sub-tasks with multiple task types, the router applies the minimum score across all required task types (the model must be capable at ALL stated types).

**Independent sub-task detection:** After building per-subtask decisions, the router performs a topological sort on the `depends_on` graph and groups tasks into execution waves. Tasks within the same wave have no dependencies on each other and can run concurrently. The waves are stored in `RoutingPlan.parallel_groups` and displayed in the plan preview. The executor runs them sequentially in v1, but the grouping is surfaced so developers can see which sub-tasks would benefit most from v2 parallel execution.

**Interface:**
```python
router = Router(registry, config)
plan: RoutingPlan = router.plan(subtasks, task_summary="Build a REST API...")
```

---

### 7.3 Plan Preview (`planner.py`)

**Responsibility:** Render the routing plan as a Rich table in the terminal and optionally as a JSON/dict for programmatic inspection.

**Terminal output format:**

```
╔═════════════════════════════════════════════════════════════════════════════╗
║  SIMPLTKNOPT ROUTING PLAN                                                  ║
║  Task: "Build a REST API for user management"                              ║
╠═══╦══════════════════════════╦═══════════════════╦═════════════╦══════════╣
║ # ║ Sub-task                 ║ Model             ║ Est. Tokens ║ Est. Cost║
╠═══╬══════════════════════════╬═══════════════════╬═════════════╬══════════╣
║   ║ ── Wave 1 (independent) ─────────────────────────────────────────── ║
║ 1 ║ Plan API routes          ║ Gemini 2.0 Flash  ║ 1.2k / 0.8k ║ $0.0005 ║
║   ║ ── Wave 2 (depends on 1) ────────────────────────────────────────── ║
║ 2 ║ Write data models        ║ DeepSeek-V3       ║ 2.0k / 3.0k ║ $0.0016 ║
║ 3 ║ Write endpoint handlers  ║ DeepSeek-V3       ║ 3.5k / 5.0k ║ $0.0026 ║
║   ║  ↳ tasks 2 & 3 can run in parallel (v2)                            ║
║   ║ ── Wave 3 (depends on 2, 3) ─────────────────────────────────────── ║
║ 4 ║ Write unit tests         ║ GPT-4o mini       ║ 3.0k / 3.0k ║ $0.0009 ║
║ 5 ║ Security review          ║ Claude Sonnet 4.6 ║ 6.0k / 1.5k ║ $0.0205 ║
║   ║  ↳ tasks 4 & 5 can run in parallel (v2)                            ║
╠═══╩══════════════════════════╩═══════════════════╩═════════════╩══════════╣
║  OPTIMIZED ESTIMATE         $0.0261                                        ║
╠════════════════════════════════════════════════════════════════════════════╣
║  vs. all Claude Opus 4.6    $0.84    (32× more expensive)                 ║
║  vs. all GPT-4o             $0.31    (12× more expensive)                 ║
║  vs. all Claude Sonnet 4.6  $0.18    (6.9× more expensive)               ║
║  vs. all Gemini 1.5 Pro     $0.11    (4.2× more expensive)               ║
╠════════════════════════════════════════════════════════════════════════════╣
║  Quality threshold: 0.75  |  Verification: off  |  Max escalations: 2    ║
╚════════════════════════════════════════════════════════════════════════════╝

  Proceed? [Y]es / [N]o / [O]verride a routing decision:
```

**Behaviors:**
- `[Y]` immediately begins execution
- `[N]` returns the `RoutingPlan` object without executing (developer can modify and re-run)
- `[O]` enters override mode: developer picks a sub-task number and types a different model ID; plan recalculates cost and re-displays

**Programmatic (non-interactive) mode:**
```python
plan = router.plan(subtasks)
print(plan.model_dump_json(indent=2))   # inspect as JSON; no prompt
result = executor.run(plan, interactive=False)
```

---

### 7.4 Executor (`executor.py`)

**Responsibility:** Run each sub-task in plan order, passing context between them, via LiteLLM.

**Key behaviors:**
- Builds a prompt per sub-task that includes: the sub-task description + the concatenated outputs of all `depends_on` sub-tasks (if `context_passthrough=True`)
- Calls `litellm.completion()` with the assigned model
- Captures actual token counts and cost via LiteLLM's usage tracking
- **Verification is opt-in per sub-task** (`SubTask.verify = True`). When disabled (the default), the output is accepted as-is and the Verifier is not called — no extra latency, no extra cost.
- When `verify=True`: passes result to Verifier; on failure escalates to `ranked_alternatives[0]`, increments `escalation_count`, re-runs
- On max escalations reached without passing: records `quality_passed=False` in result, continues to next sub-task (behavior configurable: raise exception or continue)
- Handles `tenacity` retries for transient API errors (rate limits, timeouts)

**Context management:**
- Simple concatenation of prior outputs (v1)
- Token budget management: if accumulated context exceeds 80% of assigned model's context window, summarize prior outputs using the cheapest capable summarization model before injecting

**Interface:**
```python
executor = Executor(config)
result: ExecutionResult = executor.run(plan)
```

---

### 7.5 Verifier (`verifier.py`)

**Responsibility:** Given a sub-task and its output, determine if the output meets the quality threshold.

**Verification strategy by task type:**

| Task Type          | Primary Verification Method                                   |
|--------------------|---------------------------------------------------------------|
| code_generation    | Syntax check (AST parse) + LLM judge rubric                  |
| code_review        | LLM judge: did it identify issues? are suggestions specific?  |
| reasoning          | LLM judge with chain-of-thought verification                  |
| summarization      | Coverage check: key entities from input present in output     |
| data_extraction    | Schema validation (if output_schema provided) + LLM judge    |
| classification     | Enum membership check + confidence score                      |
| translation        | Back-translation round-trip score (optional) + LLM judge     |
| qa                 | Entailment check against source document + LLM judge         |
| instruction_following | Constraint checklist: did output satisfy each instruction? |
| tool_use           | JSON schema validation + required fields present             |
| creative_writing   | LLM judge: tone/style/coherence rubric                       |

**LLM judge model:** Default is the cheapest model with `classification` score >= 0.80. Developer can override.

**Judge prompt template (configurable):**
```
You are a strict quality evaluator. The following sub-task was given to an AI model.

SUB-TASK: {subtask.description}
TASK TYPE: {task_type}
OUTPUT: {output}

Score the output from 0.0 to 1.0 on whether it fully satisfies the sub-task.
Consider: completeness, accuracy, and correctness.
Return JSON: {"score": float, "passed": bool, "reason": str}
Quality threshold for pass: {threshold}
```

**Interface:**
```python
verifier = Verifier(registry, config)
score: float = verifier.verify(subtask, output)
# returns 0.0–1.0; routing engine uses subtask.quality_threshold to determine pass/fail
```

---

## 8. Developer Interface

### 8.1 Python SDK

The primary integration surface. Designed to require no more than 5 lines to get a working pipeline.

```python
from simpltknopt import SimplTknOpt

stko = SimplTknOpt()  # reads API keys from env or stko.yaml; uses ~/.stko/preferences.yaml for defaults

# Option A: Fully automatic (uses persisted or stko.yaml threshold; verification off)
result = stko.run("Build a REST API for user management")

# Option A with one-off threshold override (does not change persisted preference)
result = stko.run("Build a REST API for user management", threshold=0.85)

# Option B: Preview plan before executing
plan = stko.plan("Build a REST API for user management")
# → prints routing table with parallel wave groups, prompts [Y/N/O]
result = stko.execute(plan)

# Option C: Full manual control with selective verification
from simpltknopt import SubTask, TaskType

subtasks = [
    SubTask(
        description="Write OpenAPI spec for /users CRUD",
        task_types=[TaskType.CODE_GENERATION],
        estimated_input_tokens=1500,
        estimated_output_tokens=2000,
        quality_threshold=0.80,
        # verify defaults to False — accepted as-is
    ),
    SubTask(
        description="Review the spec for security issues",
        task_types=[TaskType.CODE_REVIEW],
        depends_on=["write-openapi-spec"],
        quality_threshold=0.85,
        verify=True,  # opt in: this sub-task is high-stakes, judge the output
    ),
]
plan = stko.plan_from_subtasks(subtasks, task_summary="OpenAPI spec + review")
result = stko.execute(plan, interactive=False)

# Inspect results
for r in result.subtask_results:
    print(f"{r.subtask_id}: {r.model_used} | quality={r.quality_score:.2f} | cost=${r.actual_cost_usd:.4f}")
print(f"Total: ${result.total_actual_cost_usd:.4f} | Saved: ${result.total_saved_vs_plan:.4f}")
```

### 8.2 Configuration (`stko.yaml`)

```yaml
# stko.yaml — project-level config (check into version control, exclude secrets)

api_keys:
  anthropic: ${ANTHROPIC_API_KEY}
  openai: ${OPENAI_API_KEY}
  google: ${GOOGLE_API_KEY}
  deepseek: ${DEEPSEEK_API_KEY}

defaults:
  quality_threshold: 0.75       # global default; overridable per run (--threshold) or per sub-task
  verify: false                 # verification is opt-in; enable globally here or per SubTask
  max_escalations: 2            # times to retry with next-best model (only relevant when verify=true)
  decomposition_model: claude-haiku-4-5-20251001   # model used to decompose tasks
  judge_model: auto             # "auto" = cheapest model with classification >= 0.80
  interactive: true             # show plan table and prompt before executing
  on_quality_failure: continue  # "continue" or "raise"

# Registry: fetched from URL; set registry_url to self-host in air-gapped environments
registry_url: https://registry.simpltknopt.dev/models.yaml

# Optional: disable specific providers (e.g., data residency requirements)
enabled_providers:
  - anthropic
  - openai
  - google
  - deepseek

# Optional: registry overrides (add custom models or adjust scores for your domain)
registry_overrides:
  - id: deepseek/deepseek-chat
    capabilities:
      code_generation: 0.93     # you've validated this for your codebase
```

**Persisted user preferences (`~/.stko/preferences.yaml`):** The first time a developer runs `stko init` or any `stko` command, they are prompted for their preferred quality threshold (default shown: 0.75). The answer is written to `~/.stko/preferences.yaml` as `quality_threshold`. This file is machine-global (not project-specific) and takes lower precedence than a project-level `stko.yaml`. Developers can change it at any time with `stko config set quality_threshold 0.85`, or override for a single run with `--threshold 0.85`.

### 8.3 CLI (`stko`)

```bash
# First-time setup: saves preferred threshold to ~/.stko/preferences.yaml
stko init

# Preview routing plan only (no execution)
stko plan "Build a REST API for user management"

# Preview with a one-off threshold override (does not change saved preference)
stko plan "Build a REST API for user management" --threshold 0.85

# Plan + execute
stko run "Build a REST API for user management"

# Plan + execute with verification enabled for all sub-tasks this run
stko run "Build a REST API for user management" --verify

# Run non-interactively (CI/CD)
stko run "Summarize these 10 documents" --no-interactive --output result.json

# Show registered models and their scores
stko models list
stko models list --task-type code_generation   # sorted by capability score

# Force a registry refresh from the hosted URL
stko registry refresh

# Compare cost across models for a rough token estimate
stko cost-estimate --input-tokens 5000 --output-tokens 2000

# View or update persisted user preferences
stko config show
stko config set quality_threshold 0.85

# Validate your stko.yaml and API key connectivity
stko doctor
```

---

## 9. File & Project Structure

```
simpltknopt/
│
├── simpltknopt/                    # installable package
│   ├── __init__.py                 # exports: SimplTknOpt, SubTask, TaskType, etc.
│   ├── config.py                   # stko.yaml + ~/.stko/preferences.yaml loading
│   ├── decomposer.py               # Decomposer class
│   ├── registry.py                 # ModelRegistry: fetch from URL, ETag cache, overlay merge
│   ├── router.py                   # Router class + routing + parallel-group detection
│   ├── planner.py                  # RoutingPlan Rich display + override flow
│   ├── executor.py                 # Executor class, LiteLLM calls, context mgmt
│   ├── verifier.py                 # Verifier class + per-type rubrics (opt-in)
│   ├── cli.py                      # Typer CLI (stko entrypoint)
│   └── models/
│       ├── __init__.py
│       ├── task.py                 # SubTask, RoutingPlan, ExecutionResult
│       └── registry_models.py     # ModelEntry, PricingInfo, CapabilityMap
│
├── registry/
│   └── models.yaml                 # seed registry — also served at registry.simpltknopt.dev
│                                   # (NOT bundled in the installable package; fetched at runtime)
│
├── examples/
│   ├── simple_pipeline.py          # 5-line quickstart
│   ├── manual_subtasks.py          # manually defined sub-task list
│   ├── ci_noninteractive.py        # CI/CD usage without prompts
│   ├── with_verification.py        # opt-in verification on selected sub-tasks
│   └── custom_rubric.py            # custom quality verifier
│
├── tests/
│   ├── test_registry.py            # includes offline/cache fallback tests
│   ├── test_router.py              # includes parallel-group detection tests
│   ├── test_verifier.py
│   ├── test_decomposer.py
│   └── test_cli.py
│
# User machine (not in repo):
# ~/.stko/
#   preferences.yaml               # persisted user defaults (quality_threshold, etc.)
#   registry_cache.yaml            # last-fetched registry with ETag
│
├── pyproject.toml                  # package metadata, deps, [tool.stko] defaults
├── stko.yaml.example               # template config for developers to copy
└── SPEC.md                         # this document
```

---

## 10. Key Design Decisions & Trade-offs

### Why a static YAML registry, not a live benchmark?

Capability scores in the registry are curated defaults. They will be wrong for some tasks in some domains. The design acknowledges this explicitly: developers are expected to tune scores in their `stko.yaml` overlay after observing performance. A live benchmark calibration system would be more accurate but would (a) require expensive multi-model test runs before first use, and (b) create a chicken-and-egg problem. Static defaults + easy override is the right v1 posture.

### Why not just use the cheapest model for everything?

The routing engine enforces a quality floor (`quality_threshold`). A developer who sets threshold to 1.0 will always get the best available model. A developer who sets it to 0.0 will always get the cheapest. The default (0.75) is a deliberately opinionated starting point that routes most straightforward sub-tasks to cheaper models while protecting high-stakes sub-tasks (code review, security analysis) from under-qualified assignment.

### Why LiteLLM and not direct SDKs?

Direct SDKs (anthropic, openai, google-generativeai) would require 4x the integration surface and 4x the auth management. LiteLLM's unified interface is the right abstraction here. It is open source, actively maintained, and already used in production by many routing systems.

### Why sequential execution in v1, but parallel-group detection now?

Dependent sub-tasks cannot run in parallel regardless. Independent sub-tasks can, but the async harness adds significant complexity. The right posture for v1 is to detect and surface the parallelism opportunity in the routing plan (so developers can see it and plan for it) without yet executing it. When v2 adds async execution, the grouping data is already in the `RoutingPlan` object — no model changes needed.

### On quality verification accuracy

The Verifier uses a language model as a judge. This introduces a well-known limitation: the judge can be wrong. Mitigations:
- The judge is applied after each sub-task, not just at the end (failures fail fast)
- The judge prompt is task-type-specific, not generic
- Syntax/schema checks are used where deterministic validation is possible
- Developers can supply custom rubrics (strings appended to the judge prompt)
- Escalation provides a second attempt, not just a re-run of the same model

---

## 11. Non-Goals and Explicit Deferments

| Capability                              | Rationale for deferment        |
|-----------------------------------------|-------------------------------|
| Async / parallel sub-task execution     | Detection is v1; execution deferred to v2 |
| Streaming responses                     | LiteLLM supports; add in v2   |
| Offline/local decomposer model          | Not supported; all decomposition uses API calls |
| Automatic registry calibration          | Expensive; add in v2          |
| Fine-tuned model support                | Custom registry entry covers it; no native benchmarks |
| Multi-turn / agentic sub-tasks          | Out of scope for routing layer |
| Web UI                                  | CLI + SDK is sufficient for developer audience |
| Cost alerts / spend limits              | Delegate to provider dashboards in v1 |

---

## 12. Decision Log

All pre-implementation open questions are resolved.

| # | Question | Decision |
|---|---|---|
| 1 | Quality threshold default | **0.75 hard default.** Persisted to `~/.stko/preferences.yaml` on first run via `stko init`. Overridable per project in `stko.yaml`, per run with `--threshold`, or per sub-task via `SubTask.quality_threshold`. |
| 2 | Offline/local decomposer | **Not supported.** Decomposition always uses an API-backed model. Developers who want zero-cost planning can supply their own `SubTask` list and skip decomposition entirely. |
| 3 | Registry source | **Fetched from a hosted URL** (`https://registry.simpltknopt.dev/models.yaml`) with ETag caching. Last-fetched copy used as offline fallback. Self-hosting supported via `registry_url` in `stko.yaml`. |
| 4 | Verification on/off | **Opt-in per sub-task** (`SubTask.verify = True`). Off by default to avoid hidden latency and cost. Can be toggled globally for a run with `--verify` or set as default in `stko.yaml`. |
| 5 | Independent sub-task detection | **Yes — detect and surface in v1.** The router performs a topological sort and groups tasks into parallel waves, displayed in the plan preview. Sequential execution only in v1; the grouping data is stored in `RoutingPlan.parallel_groups` ready for v2 async execution. |
| 6 | Cost tiebreaker | **Quality score descending.** When two models have identical estimated cost, the one with the higher capability score for the primary task type wins. No alphabetical fallback. |

---

## 13. Implementation Sequence (for post-approval)

Once this spec is approved, implementation should proceed in this order to maximize testability:

1. `models/task.py` and `models/registry_models.py` — data models first, no logic
2. `config.py` — `stko.yaml` + `~/.stko/preferences.yaml` loading; env var resolution; precedence chain
3. `registry.py` — URL fetch with ETag caching, offline fallback, overlay merge; fully unit-testable with mocked HTTP
4. `router.py` — pure routing algorithm + parallel-group topological sort; no API calls; fully unit-testable
5. `planner.py` — Rich display with wave groupings, cost table, override flow; testable with mock plans
6. `verifier.py` — deterministic checks first (AST parse, schema validation), then LLM judge; opt-in path tested separately
7. `decomposer.py` — live API call; mock in unit tests
8. `executor.py` — live API calls; integration tests only; wires opt-in verifier path
9. `cli.py` — wires everything together; includes `stko init`, `stko config`, `stko registry refresh`
10. `examples/` — written last, validate end-to-end flow including `with_verification.py`

---

*Specification approved. All open questions resolved. Ready for implementation.*
