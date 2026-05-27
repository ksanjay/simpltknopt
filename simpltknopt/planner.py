"""Rich terminal rendering of RoutingPlan with wave groups and cost comparison."""
from __future__ import annotations

from typing import Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .models.task import RoutingDecision, RoutingPlan, SubTask

console = Console()

_PROVIDER_COLORS = {
    "anthropic": "bright_magenta",
    "openai": "bright_green",
    "google": "bright_blue",
    "deepseek": "bright_cyan",
}


def _provider_from_model(model_id: str) -> str:
    if "claude" in model_id:
        return "anthropic"
    if "gpt" in model_id or model_id.startswith("o3") or model_id.startswith("o1"):
        return "openai"
    if "gemini" in model_id:
        return "google"
    if "deepseek" in model_id:
        return "deepseek"
    return ""


def _color_model(display_name: str, model_id: str) -> Text:
    provider = _provider_from_model(model_id)
    color = _PROVIDER_COLORS.get(provider, "white")
    return Text(display_name, style=color)


def _fmt_cost(cost: float) -> str:
    if cost < 0.0001:
        return f"${cost:.6f}"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.3f}"


def _fmt_tokens(inp: int, out: int) -> str:
    def _k(n: int) -> str:
        return f"{n/1000:.1f}k" if n >= 1000 else str(n)
    return f"{_k(inp)} / {_k(out)}"


def render_plan(plan: RoutingPlan) -> None:
    """Print the full routing plan table to the terminal."""

    # Build a lookup: subtask_id → SubTask
    subtask_map: dict[str, SubTask] = {s.id: s for s in plan.subtasks}

    table = Table(
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold",
        expand=True,
        title=f"[bold]SimplTknOpt Routing Plan[/bold]\n[dim]{plan.task_summary}[/dim]",
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Sub-task", min_width=24)
    table.add_column("Model", min_width=18)
    table.add_column("Cap.", justify="right", width=5)
    table.add_column("Est. tokens", justify="right", width=13)
    table.add_column("Est. cost", justify="right", width=10)
    table.add_column("✓", width=3, justify="center")

    # Build an ordering that keeps wave groupings visible
    ordered_ids: list[str] = []
    for wave in plan.parallel_groups:
        ordered_ids.extend(wave)
    # Any subtask not in parallel groups (shouldn't happen, but safety)
    for s in plan.subtasks:
        if s.id not in ordered_ids:
            ordered_ids.append(s.id)

    global_idx = 0
    for wave_idx, wave in enumerate(plan.parallel_groups):
        # Wave separator row
        if len(plan.parallel_groups) > 1:
            wave_label = f" Wave {wave_idx + 1}"
            if len(wave) > 1:
                wave_label += f"  [dim](tasks {', '.join(wave)} can run in parallel — v2)[/dim]"
            table.add_row(
                "",
                Text(f"── {wave_label} ──────────────────────", style="dim"),
                "", "", "", "", "",
                end_section=False,
            )

        for sid in wave:
            global_idx += 1
            subtask = subtask_map.get(sid)
            decision = plan.decision_for(sid)
            if not subtask or not decision:
                continue

            verify_mark = "[green]✓[/green]" if decision.verify else ""
            override_suffix = (
                f" [dim](forced)[/dim]" if decision.override_reason == "forced_model" else ""
            )

            table.add_row(
                str(global_idx),
                subtask.description[:52] + ("…" if len(subtask.description) > 52 else ""),
                Text.assemble(
                    _color_model(decision.assigned_model_display, decision.assigned_model),
                    override_suffix,
                ),
                f"{decision.capability_score:.0%}",
                _fmt_tokens(subtask.estimated_input_tokens, subtask.estimated_output_tokens),
                _fmt_cost(decision.estimated_cost_usd),
                verify_mark,
            )

    console.print(table)

    # ── Summary footer ──────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(
        f"  [bold]Optimized estimate:[/bold]  [green]{_fmt_cost(plan.total_estimated_cost_usd)}[/green]"
    )

    if plan.comparison:
        sorted_comp = sorted(plan.comparison.items(), key=lambda x: x[1], reverse=True)
        for model_name, cost in sorted_comp:
            if cost > plan.total_estimated_cost_usd:
                ratio = cost / max(plan.total_estimated_cost_usd, 0.000001)
                lines.append(
                    f"  [dim]vs. all {model_name:<22}[/dim]  "
                    f"[yellow]{_fmt_cost(cost)}[/yellow]  "
                    f"[dim]({ratio:.1f}× more expensive)[/dim]"
                )

    lines.append("")
    lines.append(
        f"  [dim]Quality threshold: {plan.quality_threshold:.2f}  |  "
        f"Verification: {'on' if plan.verify_enabled else 'off'}[/dim]"
    )

    console.print(Panel("\n".join(lines), box=box.SIMPLE))


def prompt_proceed(plan: RoutingPlan) -> tuple[bool, Optional[RoutingPlan]]:
    """
    Interactive prompt: [Y]es / [N]o / [O]verride.
    Returns (should_execute, possibly_modified_plan).
    """
    while True:
        choice = Prompt.ask(
            "\n  Proceed?",
            choices=["y", "n", "o"],
            default="y",
            show_choices=True,
        ).lower()

        if choice == "n":
            console.print("[dim]Routing plan returned without executing.[/dim]")
            return False, plan

        if choice == "y":
            return True, plan

        # Override mode
        plan = _interactive_override(plan)
        render_plan(plan)


def _interactive_override(plan: RoutingPlan) -> RoutingPlan:
    """Allow developer to reassign a model for a specific sub-task."""
    subtask_map = {s.id: s for s in plan.subtasks}

    console.print("\n[bold]Override mode[/bold] — enter sub-task number to reassign:")
    idx_str = Prompt.ask("  Sub-task #")
    try:
        idx = int(idx_str) - 1
        # Build ordered list matching what was rendered
        ordered: list[str] = []
        for wave in plan.parallel_groups:
            ordered.extend(wave)
        sid = ordered[idx]
    except (ValueError, IndexError):
        console.print("[red]Invalid selection.[/red]")
        return plan

    subtask = subtask_map.get(sid)
    if not subtask:
        console.print("[red]Sub-task not found.[/red]")
        return plan

    console.print(f"  Reassigning: [bold]{subtask.description}[/bold]")
    new_model = Prompt.ask("  New model ID (e.g. claude-sonnet-4-6)")

    # Rebuild decisions with override applied
    new_decisions = []
    for d in plan.decisions:
        if d.subtask_id == sid:
            from .registry import get_registry
            entry = get_registry().get(new_model)
            cost = (
                entry.estimated_cost(
                    subtask.estimated_input_tokens, subtask.estimated_output_tokens
                )
                if entry
                else d.estimated_cost_usd
            )
            new_decisions.append(
                d.model_copy(
                    update={
                        "assigned_model": new_model,
                        "assigned_model_display": entry.display_name if entry else new_model,
                        "estimated_cost_usd": cost,
                        "override_reason": "developer_override",
                    }
                )
            )
        else:
            new_decisions.append(d)

    new_total = sum(d.estimated_cost_usd for d in new_decisions)
    return plan.model_copy(
        update={
            "decisions": new_decisions,
            "total_estimated_cost_usd": new_total,
        }
    )
