"""Typer CLI — `stko` entrypoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

app = typer.Typer(
    name="stko",
    help="SimplTknOpt — token-optimized model router.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
console = Console()

# Sub-command groups
models_app = typer.Typer(help="Inspect the model registry.")
config_app = typer.Typer(help="View and update persisted preferences.")
registry_app = typer.Typer(help="Manage the model registry cache.")

app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")
app.add_typer(registry_app, name="registry")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_stack():
    from .config import get_config
    from .registry import get_registry
    from .router import Router
    from .decomposer import Decomposer
    from .executor import Executor
    from .verifier import Verifier

    cfg = get_config()
    reg = get_registry(config=cfg)
    router = Router(registry=reg, config=cfg)
    decomposer = Decomposer(config=cfg)
    verifier = Verifier(registry=reg, config=cfg)
    executor = Executor(registry=reg, config=cfg, verifier=verifier)
    return cfg, reg, router, decomposer, executor


# ── stko init ─────────────────────────────────────────────────────────────────

@app.command()
def init():
    """First-time setup: save preferred quality threshold to ~/.stko/preferences.yaml."""
    from .config import get_config, PREFS_PATH

    cfg = get_config()
    console.print("\n[bold]SimplTknOpt Setup[/bold]\n")

    current = cfg.quality_threshold
    raw = Prompt.ask(
        f"  Quality threshold (0.0–1.0)",
        default=str(current),
    )
    try:
        threshold = float(raw)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError
    except ValueError:
        console.print("[red]Invalid value. Using default 0.75.[/red]")
        threshold = 0.75

    cfg.save_preference("quality_threshold", threshold)
    console.print(f"\n  [green]✓[/green] Saved quality_threshold={threshold} to {PREFS_PATH}")
    console.print("  Run [bold]stko doctor[/bold] to verify your API keys.\n")


# ── stko plan ─────────────────────────────────────────────────────────────────

@app.command()
def plan(
    task: Annotated[str, typer.Argument(help="Task description to decompose and route")],
    threshold: Annotated[Optional[float], typer.Option("--threshold", "-t", help="Override quality threshold for this run")] = None,
    verify: Annotated[bool, typer.Option("--verify/--no-verify", help="Enable verification for all sub-tasks")] = False,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Write routing plan JSON to file")] = None,
):
    """Decompose a task and preview the routing plan without executing."""
    from .planner import render_plan

    cfg, reg, router, decomposer, _ = _get_stack()

    with console.status("[cyan]Decomposing task…[/cyan]"):
        try:
            subtasks = decomposer.decompose(task)
        except Exception as e:
            console.print(f"[red]Decomposition failed:[/red] {e}")
            raise typer.Exit(1)

    routing_plan = router.plan(
        subtasks,
        task_summary=task,
        run_threshold=threshold,
        run_verify=verify or None,
    )

    render_plan(routing_plan)

    if output:
        output.write_text(routing_plan.model_dump_json(indent=2))
        console.print(f"\n  Plan saved to [cyan]{output}[/cyan]")


# ── stko run ──────────────────────────────────────────────────────────────────

@app.command()
def run(
    task: Annotated[str, typer.Argument(help="Task description to execute")],
    threshold: Annotated[Optional[float], typer.Option("--threshold", "-t")] = None,
    verify: Annotated[bool, typer.Option("--verify/--no-verify")] = False,
    interactive: Annotated[bool, typer.Option("--interactive/--no-interactive")] = True,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Write ExecutionResult JSON to file")] = None,
):
    """Decompose, route, and execute a task."""
    from .planner import render_plan, prompt_proceed

    cfg, reg, router, decomposer, executor = _get_stack()

    with console.status("[cyan]Decomposing task…[/cyan]"):
        try:
            subtasks = decomposer.decompose(task)
        except Exception as e:
            console.print(f"[red]Decomposition failed:[/red] {e}")
            raise typer.Exit(1)

    routing_plan = router.plan(
        subtasks,
        task_summary=task,
        run_threshold=threshold,
        run_verify=verify or None,
    )

    render_plan(routing_plan)

    should_run = True
    if interactive and cfg.interactive:
        should_run, routing_plan = prompt_proceed(routing_plan)

    if not should_run:
        raise typer.Exit(0)

    result = executor.run(routing_plan)

    # Print summary
    console.print(f"\n[bold green]✓ Done[/bold green]  "
                  f"Total cost: [green]${result.total_actual_cost_usd:.4f}[/green]  "
                  f"Saved vs flagship: [dim]${result.total_saved_vs_flagship:.4f}[/dim]\n")

    for r in result.subtask_results:
        esc = f"  [yellow](escalated ×{r.escalation_count})[/yellow]" if r.escalation_count else ""
        qual = f"  quality={r.quality_score:.2f}" if r.verified else ""
        console.print(f"  [dim]{r.subtask_id}[/dim]  {r.model_used}  "
                      f"${r.actual_cost_usd:.4f}{qual}{esc}")

    if output:
        output.write_text(result.model_dump_json(indent=2))
        console.print(f"\n  Result saved to [cyan]{output}[/cyan]")


# ── stko models list ──────────────────────────────────────────────────────────

@models_app.command("list")
def models_list(
    task_type: Annotated[Optional[str], typer.Option("--task-type", "-t", help="Filter/sort by task type")] = None,
    provider: Annotated[Optional[str], typer.Option("--provider", "-p")] = None,
):
    """List all registered models with capability scores and pricing."""
    from .registry import get_registry
    from .models.task import TaskType

    reg = get_registry()
    all_models = reg.all_models()

    if provider:
        all_models = [m for m in all_models if m.provider == provider]

    tt = None
    if task_type:
        try:
            tt = TaskType(task_type)
            all_models.sort(key=lambda m: m.capability_score(tt), reverse=True)
        except ValueError:
            console.print(f"[red]Unknown task type: {task_type}[/red]")
            raise typer.Exit(1)
    else:
        all_models.sort(key=lambda m: m.pricing.input_per_mtok)

    table = Table(title="Model Registry", show_lines=False)
    table.add_column("Model", min_width=22)
    table.add_column("Provider", width=10)
    table.add_column("In $/Mtok", justify="right", width=10)
    table.add_column("Out $/Mtok", justify="right", width=11)
    table.add_column("Context", justify="right", width=9)
    if tt:
        table.add_column(f"{task_type}", justify="right", width=8)

    for m in all_models:
        ctx = f"{m.context_window // 1000}k"
        row = [
            m.display_name,
            m.provider,
            f"${m.pricing.input_per_mtok:.2f}",
            f"${m.pricing.output_per_mtok:.2f}",
            ctx,
        ]
        if tt:
            row.append(f"{m.capability_score(tt):.0%}")
        table.add_row(*row)

    console.print(table)


# ── stko cost-estimate ────────────────────────────────────────────────────────

@app.command("cost-estimate")
def cost_estimate(
    input_tokens: Annotated[int, typer.Option("--input-tokens", "-i")] = 1000,
    output_tokens: Annotated[int, typer.Option("--output-tokens", "-o")] = 500,
):
    """Compare cost of a fixed token budget across all registered models."""
    from .registry import get_registry

    reg = get_registry()
    models = sorted(reg.all_models(), key=lambda m: m.estimated_cost(input_tokens, output_tokens))

    table = Table(title=f"Cost for {input_tokens} input + {output_tokens} output tokens")
    table.add_column("Model", min_width=22)
    table.add_column("Provider", width=10)
    table.add_column("Cost", justify="right", width=10)

    for m in models:
        cost = m.estimated_cost(input_tokens, output_tokens)
        table.add_row(m.display_name, m.provider, f"${cost:.5f}")

    console.print(table)


# ── stko config show / set ────────────────────────────────────────────────────

@config_app.command("show")
def config_show():
    """Show current effective configuration."""
    from .config import get_config, PREFS_PATH

    cfg = get_config()
    table = Table(title="Effective Configuration", show_header=True)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_column("Source", style="dim")

    rows = [
        ("quality_threshold", cfg.quality_threshold),
        ("verify", cfg.verify),
        ("max_escalations", cfg.max_escalations),
        ("decomposition_model", cfg.decomposition_model),
        ("judge_model", cfg.judge_model),
        ("interactive", cfg.interactive),
        ("on_quality_failure", cfg.on_quality_failure),
        ("registry_url", cfg.registry_url),
        ("enabled_providers", ", ".join(cfg.enabled_providers)),
    ]
    for key, val in rows:
        source = "prefs" if key in cfg._prefs else ("stko.yaml" if key in cfg._project.get("defaults", {}) else "default")
        table.add_row(key, str(val), source)

    console.print(table)
    console.print(f"\n  Preferences file: [dim]{PREFS_PATH}[/dim]")


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Preference key (e.g. quality_threshold)")],
    value: Annotated[str, typer.Argument(help="New value")],
):
    """Persist a user preference to ~/.stko/preferences.yaml."""
    from .config import get_config

    cfg = get_config()
    # Coerce to correct type
    coerced: object = value
    if value.lower() in ("true", "false"):
        coerced = value.lower() == "true"
    else:
        try:
            coerced = float(value) if "." in value else int(value)
        except ValueError:
            pass

    cfg.save_preference(key, coerced)
    console.print(f"  [green]✓[/green] {key} = {coerced}")


# ── stko registry refresh ─────────────────────────────────────────────────────

@registry_app.command("refresh")
def registry_refresh():
    """Force a fresh fetch of the model registry from the hosted URL."""
    from .config import get_config
    from .registry import get_registry

    cfg = get_config()
    console.print(f"  Fetching registry from [cyan]{cfg.registry_url}[/cyan]…")
    try:
        get_registry(config=cfg, force_refresh=True)
        console.print("  [green]✓[/green] Registry updated.")
    except Exception as e:
        console.print(f"  [red]✗[/red] Fetch failed: {e}")
        raise typer.Exit(1)


# ── stko doctor ───────────────────────────────────────────────────────────────

@app.command()
def doctor():
    """Validate configuration, API keys, and registry connectivity."""
    from .config import get_config
    from .registry import get_registry

    cfg = get_config()
    all_ok = True

    console.print("\n[bold]SimplTknOpt Doctor[/bold]\n")

    # API keys
    providers = ["anthropic", "openai", "google", "deepseek"]
    for p in providers:
        key = cfg.get_api_key(p)
        if key:
            masked = key[:8] + "…" if len(key) > 8 else "***"
            console.print(f"  [green]✓[/green] {p:<12} key found ({masked})")
        else:
            console.print(f"  [yellow]–[/yellow] {p:<12} no key configured")

    configured = cfg.configured_providers()
    if not configured:
        console.print("\n  [red]✗[/red] No API keys found. Set env vars or add to stko.yaml.")
        all_ok = False

    # Registry
    console.print()
    try:
        reg = get_registry(config=cfg)
        count = len(reg.all_models())
        console.print(f"  [green]✓[/green] Registry loaded  ({count} models)")
    except Exception as e:
        console.print(f"  [red]✗[/red] Registry error: {e}")
        all_ok = False

    # Preferences file
    from .config import PREFS_PATH
    console.print()
    if PREFS_PATH.exists():
        console.print(f"  [green]✓[/green] Preferences file  {PREFS_PATH}")
    else:
        console.print(f"  [yellow]–[/yellow] No preferences file. Run [bold]stko init[/bold] to create one.")

    console.print(f"\n  Quality threshold: [bold]{cfg.quality_threshold}[/bold]")
    console.print(f"  Configured providers: [bold]{', '.join(configured) or 'none'}[/bold]\n")

    if not all_ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
