"""Quality verification — opt-in per sub-task. Uses a judge model + deterministic checks."""
from __future__ import annotations

import ast
import json
import re
from typing import Optional

from .config import Config, get_config
from .models.registry_models import ModelEntry
from .models.task import SubTask, TaskType
from .registry import ModelRegistry, get_registry

_JUDGE_PROMPT = """\
You are a strict quality evaluator. An AI model was given the following sub-task and produced an output.

SUB-TASK: {description}
TASK TYPE: {task_type}
OUTPUT:
{output}

Score the output from 0.0 to 1.0 on whether it fully satisfies the sub-task.
Criteria: completeness, accuracy, and correctness for the stated task type.
Quality threshold for PASS: {threshold}

Respond with ONLY valid JSON:
{{"score": <float 0.0-1.0>, "passed": <bool>, "reason": "<one sentence>"}}
"""


class VerificationResult:
    def __init__(self, score: float, passed: bool, reason: str) -> None:
        self.score = score
        self.passed = passed
        self.reason = reason

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"VerificationResult({status}, score={self.score:.2f}, reason={self.reason!r})"


class Verifier:
    """Opt-in quality judge. Called only when SubTask.verify is True."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        config: Optional[Config] = None,
        custom_rubric: Optional[str] = None,
    ) -> None:
        self._registry = registry or get_registry()
        self._config = config or get_config()
        self._custom_rubric = custom_rubric
        self._judge_model: Optional[str] = None

    def verify(self, subtask: SubTask, output: str) -> VerificationResult:
        """Run verification for a single sub-task output."""
        threshold = (
            subtask.quality_threshold
            if subtask.quality_threshold is not None
            else self._config.quality_threshold
        )

        # 1. Deterministic checks first (fast, free)
        det = self._deterministic_check(subtask, output)
        if det is not None and not det.passed:
            return det

        # 2. LLM judge
        return self._llm_judge(subtask, output, threshold)

    # ── Deterministic checks ───────────────────────────────────────────────────

    def _deterministic_check(
        self, subtask: SubTask, output: str
    ) -> Optional[VerificationResult]:
        """Return a result if a deterministic check is conclusive, else None."""
        primary = subtask.task_types[0]

        if primary == TaskType.CODE_GENERATION:
            return self._check_syntax(output)

        if primary == TaskType.TOOL_USE:
            return self._check_json(output)

        return None

    def _check_syntax(self, output: str) -> Optional[VerificationResult]:
        """Try to parse Python code blocks in the output."""
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", output, re.DOTALL)
        if not code_blocks:
            return None  # No code fence — let LLM judge
        for block in code_blocks:
            try:
                ast.parse(block)
            except SyntaxError as e:
                return VerificationResult(
                    score=0.2,
                    passed=False,
                    reason=f"Python syntax error in code block: {e}",
                )
        return None  # Syntax OK — still run LLM for quality

    def _check_json(self, output: str) -> Optional[VerificationResult]:
        """Try to parse JSON in the output."""
        try:
            stripped = output.strip()
            # Handle ```json fences
            match = re.search(r"```(?:json)?\n(.*?)```", stripped, re.DOTALL)
            if match:
                stripped = match.group(1)
            json.loads(stripped)
            return None  # Valid JSON — let LLM assess quality
        except json.JSONDecodeError as e:
            return VerificationResult(
                score=0.1,
                passed=False,
                reason=f"Invalid JSON output: {e}",
            )

    # ── LLM judge ─────────────────────────────────────────────────────────────

    def _llm_judge(
        self, subtask: SubTask, output: str, threshold: float
    ) -> VerificationResult:
        import litellm  # lazy import

        judge = self._resolve_judge_model()
        prompt = _JUDGE_PROMPT.format(
            description=subtask.description,
            task_type=subtask.task_types[0].value,
            output=output[:4000],  # cap to avoid huge judge prompts
            threshold=threshold,
        )
        if self._custom_rubric:
            prompt += f"\n\nAdditional rubric:\n{self._custom_rubric}"

        try:
            resp = litellm.completion(
                model=judge,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
            # Extract JSON even if surrounded by prose
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = float(data.get("score", 0.0))
                passed = bool(data.get("passed", score >= threshold))
                reason = str(data.get("reason", ""))
                return VerificationResult(score=score, passed=passed, reason=reason)
        except Exception as e:
            # Verification error — fail open (don't block pipeline)
            return VerificationResult(
                score=-1.0,
                passed=True,
                reason=f"Verifier error (failing open): {e}",
            )

        return VerificationResult(score=0.5, passed=True, reason="Could not parse judge response")

    def _resolve_judge_model(self) -> str:
        if self._judge_model:
            return self._judge_model

        judge_config = self._config.judge_model
        if judge_config != "auto":
            self._judge_model = judge_config
            return self._judge_model

        # Auto: cheapest model with classification >= 0.80 from configured providers
        candidates = self._registry.models_for_providers(self._config.enabled_providers)
        eligible = [
            m for m in candidates
            if m.capability_score(TaskType.CLASSIFICATION) >= 0.80
        ]
        if eligible:
            # Sort by cost of a small classification call (500 in / 256 out)
            eligible.sort(key=lambda m: m.estimated_cost(500, 256))
            self._judge_model = eligible[0].id
        else:
            self._judge_model = self._config.decomposition_model

        return self._judge_model
