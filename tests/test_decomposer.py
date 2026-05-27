"""Tests for Decomposer: JSON parsing and fallback behaviour."""
from unittest.mock import MagicMock, patch

import pytest

from simpltknopt.config import Config
from simpltknopt.decomposer import Decomposer
from simpltknopt.models.task import SubTask, TaskType


def _cfg() -> Config:
    cfg = MagicMock(spec=Config)
    cfg.decomposition_model = "claude-haiku-4-5-20251001"
    return cfg


def _mock_response(content: str):
    resp = MagicMock()
    resp.choices[0].message.content = content
    return resp


class TestDecomposerParsing:
    def test_parses_valid_array(self):
        d = Decomposer(config=_cfg())
        raw = '''[
            {
                "id": "write-code",
                "description": "Write the main function",
                "task_types": ["code_generation"],
                "estimated_input_tokens": 500,
                "estimated_output_tokens": 800,
                "depends_on": []
            }
        ]'''
        result = d._parse(raw)
        assert len(result) == 1
        assert result[0].id == "write-code"
        assert result[0].task_types == [TaskType.CODE_GENERATION]

    def test_parses_json_in_markdown_fence(self):
        d = Decomposer(config=_cfg())
        raw = '''Here are the subtasks:
```json
[{"id": "t1", "description": "Task one", "task_types": ["summarization"],
  "estimated_input_tokens": 500, "estimated_output_tokens": 300, "depends_on": []}]
```'''
        result = d._parse(raw)
        assert len(result) == 1

    def test_invalid_task_type_defaults_to_instruction_following(self):
        d = Decomposer(config=_cfg())
        raw = '[{"id": "t1", "description": "Do something", "task_types": ["nonexistent_type"], "estimated_input_tokens": 500, "estimated_output_tokens": 300, "depends_on": []}]'
        result = d._parse(raw)
        assert result[0].task_types == [TaskType.INSTRUCTION_FOLLOWING]

    def test_depends_on_validated_against_prior_ids(self):
        d = Decomposer(config=_cfg())
        raw = '''[
            {"id": "a", "description": "A", "task_types": ["summarization"],
             "estimated_input_tokens": 500, "estimated_output_tokens": 300, "depends_on": []},
            {"id": "b", "description": "B", "task_types": ["summarization"],
             "estimated_input_tokens": 500, "estimated_output_tokens": 300, "depends_on": ["a"]},
            {"id": "c", "description": "C", "task_types": ["summarization"],
             "estimated_input_tokens": 500, "estimated_output_tokens": 300,
             "depends_on": ["a", "nonexistent"]}
        ]'''
        result = d._parse(raw)
        assert result[2].depends_on == ["a"]  # "nonexistent" stripped

    def test_fallback_on_invalid_json(self):
        d = Decomposer(config=_cfg())
        result = d._parse("This is not JSON at all")
        assert len(result) == 1
        assert result[0].task_types == [TaskType.INSTRUCTION_FOLLOWING]

    def test_decompose_calls_litellm(self):
        d = Decomposer(config=_cfg())
        valid_response = '[{"id": "t1", "description": "Write code", "task_types": ["code_generation"], "estimated_input_tokens": 500, "estimated_output_tokens": 800, "depends_on": []}]'

        with patch("litellm.completion", return_value=_mock_response(valid_response)):
            result = d.decompose("Build an API")

        assert len(result) == 1
        assert result[0].task_types == [TaskType.CODE_GENERATION]
