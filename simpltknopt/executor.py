"""Execution harness: run a RoutingPlan sequentially via LiteLLM with context passing."""
from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config, get_config
from .models.task import ExecutionResult, RoutingPlan, SubTask, SubTaskResult
from .registry import ModelRegistry, get_registry
from .verifier import Verifier

console = Console()

_CONTEXT_HEADER = """\
--- Context from previous sub-tasks ---
{context}
--- End context ---

"""

_SUMMARIZE_PROMPT = """\
Summarize the following content concisely, preserving all key outputs, decisions, and facts.
Keep it under {max_tokens} tokens.

{content}
"""


class Executor:
    """Runs a RoutingPlan, calling each model via LiteLLM in wave order."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        config: Optional[Config] = None,
        verifier: Optional[Verifier] = None,
    ) -> None:
        self._registry = registry or get_registry()
        self._config = config or get_config()
        self._verifier = verifier or Verifier(registry=self._registry, config=self._config)

    def run(
        self,
        plan: RoutingPlan,
        show_progress: bool = True,
    ) -> ExecutionResult:
        """Execute every sub-task in the plan sequentially (wave order)."""
        import litellm

        results: list[SubTaskResult] = []
        context_store: dict[str, str] = {}  # subtask_id → output text

        # Flatten waves into execution order
        execution_order: list[str] = []
        for wave in plan.parallel_groups:
            execution_order.extend(wave)
        # Fallback: any subtask not in groups (shouldn't happen)
        for s in plan.subtasks:
            if s.id not in execution_order:
                execution_order.append(s.id)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            disable=not show_progress,
        ) as progress:
            for sid in execution_order:
                subtask = plan.subtask_for(sid)
                decision = plan.decision_for(sid)
                if not subtask or not decision:
                    continue

                prog_task = progress.add_task(
                    f"[cyan]{decision.assigned_model_display}[/cyan]  {subtask.description[:50]}",
                    total=None,
                )

                result = self._run_subtask(
                    subtask=subtask,
                    decision_model=decision.assigned_model,
                    ranked_alternatives=decision.ranked_alternatives,
                    context_store=context_store,
                    do_verify=decision.verify,
                    plan=plan,
                )
                results.append(result)
                if result.output:
                    context_store[sid] = result.output

                progress.remove_task(prog_task)

        total_cost = sum(r.actual_cost_usd for r in results)

        # Cost that would have been spent on the most expensive comparison model
        flagship_cost = max(plan.comparison.values()) if plan.comparison else total_cost
        saved = max(0.0, flagship_cost - total_cost)

        return ExecutionResult(
            task_summary=plan.task_summary,
            subtask_results=results,
            total_actual_cost_usd=total_cost,
            total_saved_vs_flagship=saved,
            routing_plan=plan,
        )

    # ── Sub-task execution with escalation ────────────────────────────────────

    def _run_subtask(
        self,
        subtask: SubTask,
        decision_model: str,
        ranked_alternatives: list[str],
        context_store: dict[str, str],
        do_verify: bool,
        plan: RoutingPlan,
    ) -> SubTaskResult:
        model_queue = [decision_model] + list(ranked_alternatives)
        escalation_count = 0
        last_result: Optional[SubTaskResult] = None

        for model_id in model_queue[: self._config.max_escalations + 1]:
            result = self._call_model(subtask, model_id, context_store)

            if do_verify:
                vr = self._verifier.verify(subtask, result.output)
                result = result.model_copy(
                    update={
                        "quality_score": vr.score,
                        "quality_passed": vr.passed,
                        "verified": True,
                        "escalation_count": escalation_count,
                    }
                )
                if vr.passed:
                    return result
                escalation_count += 1
                last_result = result
                console.print(
                    f"  [yellow]⚡ Escalating[/yellow] "
                    f"[dim]{subtask.id}[/dim] "
                    f"(score {vr.score:.2f} < threshold, reason: {vr.reason})"
                )
            else:
                return result.model_copy(
                    update={"escalation_count": escalation_count}
                )

        # Max escalations reached
        if last_result is not None:
            if self._config.on_quality_failure == "raise":
                raise RuntimeError(
                    f"Sub-task '{subtask.id}' failed quality check after "
                    f"{escalation_count} escalation(s). Last score: {last_result.quality_score:.2f}"
                )
            console.print(
                f"  [red]✗ Quality failure[/red] on '{subtask.id}' "
                f"after {escalation_count} escalation(s) — continuing."
            )
            return last_result.model_copy(update={"quality_passed": False})

        # Should not reach here
        return self._call_model(subtask, decision_model, context_store)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_model(
        self,
        subtask: SubTask,
        model_id: str,
        context_store: dict[str, str],
    ) -> SubTaskResult:
        import litellm

        prompt = self._build_prompt(subtask, model_id, context_store)

        resp = litellm.completion(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        output = resp.choices[0].message.content or ""
        usage = resp.usage or {}

        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)

        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            # Fallback: estimate from registry
            entry = self._registry.get(model_id)
            cost = entry.estimated_cost(input_tokens, output_tokens) if entry else 0.0

        return SubTaskResult(
            subtask_id=subtask.id,
            model_used=model_id,
            output=output,
            actual_cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # ── Prompt building with context ──────────────────────────────────────────

    def _build_prompt(
        self,
        subtask: SubTask,
        model_id: str,
        context_store: dict[str, str],
    ) -> str:
        context_parts: list[str] = []

        if subtask.context_passthrough and subtask.depends_on:
            for dep_id in subtask.depends_on:
                if dep_id in context_store:
                    context_parts.append(f"[{dep_id}]\n{context_store[dep_id]}")

        if context_parts:
            ctx_text = "\n\n".join(context_parts)
            # Trim context if it would overflow the model's context window
            entry = self._registry.get(model_id)
            if entry:
                ctx_text = self._maybe_summarize(ctx_text, entry.context_window, model_id)
            prompt = _CONTEXT_HEADER.format(context=ctx_text) + subtask.description
        else:
            prompt = subtask.description

        return prompt

    def _maybe_summarize(self, context: str, context_window: int, model_id: str) -> str:
        """Summarize context if it would exceed 80% of the model's context window."""
        import litellm

        try:
            token_count = litellm.token_counter(model=model_id, text=context)
        except Exception:
            token_count = len(context) // 4  # rough estimate

        budget = int(context_window * 0.80)
        if token_count <= budget:
            return context

        max_out = budget // 2
        summary_prompt = _SUMMARIZE_PROMPT.format(max_tokens=max_out, content=context[:8000])
        entry = self._registry.get(model_id)
        summarize_model = self._config.decomposition_model

        try:
            resp = litellm.completion(
                model=summarize_model,
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.0,
                max_tokens=max_out,
            )
            return resp.choices[0].message.content or context
        except Exception:
            return context
