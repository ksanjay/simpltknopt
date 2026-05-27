"""Task decomposer: uses a fast model to break a task string into typed SubTasks."""
from __future__ import annotations

import json
import re
from typing import Optional

from .config import Config, get_config
from .models.task import SubTask, TaskType

_SYSTEM_PROMPT = """\
You are a task decomposition assistant for an LLM routing system.
Break the given task into a flat list of sub-tasks. Each sub-task must be:
- A single, atomic unit of work that one LLM call can complete
- Tagged with the most relevant task type(s) from the allowed list
- Given rough token estimates (input = context you'll send, output = expected response length)
- Linked to any sub-tasks it depends on by their id

Allowed task types:
code_generation, code_review, reasoning, summarization, data_extraction,
creative_writing, classification, translation, qa, instruction_following, tool_use

Respond with ONLY a JSON array. Each element:
{
  "id": "<kebab-case-slug>",
  "description": "<what this sub-task does, ≤80 chars>",
  "task_types": ["<primary>"],
  "estimated_input_tokens": <int>,
  "estimated_output_tokens": <int>,
  "depends_on": ["<id-of-upstream-subtask>"]
}

Rules:
- 2–8 sub-tasks for most tasks (no over-decomposition)
- If the task is simple (single clear request), return a 1-element array
- depends_on must only reference ids earlier in the array
- estimated_input_tokens includes context from upstream tasks if depends_on is set
"""


class Decomposer:
    """Decomposes a free-form task string into a list of SubTask objects."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or get_config()

    def decompose(self, task: str) -> list[SubTask]:
        """Call the decomposition model and return parsed SubTask list."""
        import litellm

        model = self._config.decomposition_model
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Task to decompose:\n{task}"},
        ]

        resp = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        raw = resp.choices[0].message.content.strip()
        return self._parse(raw)

    def _parse(self, raw: str) -> list[SubTask]:
        """Parse JSON array from model response into SubTask list."""
        # Strip markdown fences if present
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return self._fallback(raw)

        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return self._fallback(raw)

        subtasks: list[SubTask] = []
        seen_ids: set[str] = set()

        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                raw_types = item.get("task_types", ["instruction_following"])
                task_types = []
                for t in raw_types:
                    try:
                        task_types.append(TaskType(t))
                    except ValueError:
                        pass
                if not task_types:
                    task_types = [TaskType.INSTRUCTION_FOLLOWING]

                # Validate depends_on references
                depends_on = [d for d in item.get("depends_on", []) if d in seen_ids]

                subtask = SubTask(
                    id=item.get("id", ""),
                    description=item.get("description", ""),
                    task_types=task_types,
                    estimated_input_tokens=int(item.get("estimated_input_tokens", 1000)),
                    estimated_output_tokens=int(item.get("estimated_output_tokens", 500)),
                    depends_on=depends_on,
                )
                subtasks.append(subtask)
                seen_ids.add(subtask.id)
            except Exception:
                continue

        return subtasks if subtasks else self._fallback(raw)

    def _fallback(self, original_task: str) -> list[SubTask]:
        """If decomposition fails, return a single passthrough sub-task."""
        return [
            SubTask(
                description=original_task[:80],
                task_types=[TaskType.INSTRUCTION_FOLLOWING],
                estimated_input_tokens=1000,
                estimated_output_tokens=1000,
            )
        ]
