"""Lesson distillation from solution improvements and debug experiences."""

from __future__ import annotations

import logging
import re

from mars.llm import LLMClient
from mars.memory.lesson_pool import Lesson
from mars.prompts.debug_lesson import format_prompt as format_debug_prompt
from mars.prompts.solution_lesson import format_prompt as format_solution_prompt

logger = logging.getLogger(__name__)


def distill_solution_lesson(llm: LLMClient, best_solution: str, new_solution: str) -> Lesson | None:
    """Distill a solution improvement lesson by comparing solutions."""
    prompt = format_solution_prompt(best_solution=best_solution, new_solution=new_solution)
    try:
        result = llm.call(prompt)
        lesson_id = f"solution_lesson_{_extract_node_id(new_solution)}"
        return Lesson(id=lesson_id, category="solution", description=result)
    except Exception:
        logger.warning("Failed to distill solution lesson", exc_info=True)
        return None


def distill_debug_lesson(
    llm: LLMClient,
    source_files: str,
    source_exec_result: str,
    error_analysis: str,
    diff: str,
    final_exec_result: str,
) -> Lesson | None:
    """Distill a debugging lesson from a debug attempt."""
    prompt = format_debug_prompt(
        source_files=source_files,
        source_exec_result=source_exec_result,
        source_error_analysis=error_analysis,
        diff=diff,
        final_exec_result=final_exec_result,
    )
    try:
        result = llm.call(prompt)
        # Use incrementing ID
        lesson_id = f"debug_lesson_{hash(result) % 10000}"
        return Lesson(id=lesson_id, category="debug", description=result)
    except Exception:
        logger.warning("Failed to distill debug lesson", exc_info=True)
        return None


def _extract_node_id(text: str) -> str:
    """Try to extract node ID from solution text."""
    match = re.search(r"node_\d+", text)
    return match.group(0) if match else "unknown"
