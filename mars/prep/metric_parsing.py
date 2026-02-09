"""Metric parsing utilities for extracting metric name and direction from task descriptions."""

from __future__ import annotations

import logging

from mars.llm import LLMClient
from mars.prompts.metric_parsing import format_prompt

logger = logging.getLogger(__name__)


def parse_metric(llm: LLMClient, task_description: str) -> dict:
    """Extract metric name and direction from task description. Returns {metric_name: str, lower_is_better: bool}."""
    prompt = format_prompt(task_description=task_description)
    result = llm.call_json(prompt)
    logger.info("Parsed metric: %s (lower_is_better=%s)", result.get("metric_name"), result.get("lower_is_better"))
    return result
