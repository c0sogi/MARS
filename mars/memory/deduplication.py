"""Semantic deduplication of lessons using LLM."""

from __future__ import annotations

import logging

from mars.llm import LLMClient
from mars.memory.lesson_pool import Lesson
from mars.prompts.lesson_dedup import format_prompt

logger = logging.getLogger(__name__)


def is_duplicate_lesson(llm: LLMClient, new_lesson: Lesson, existing_lessons: list[Lesson]) -> bool:
    """Check if a new lesson is semantically duplicate of existing ones."""
    if not existing_lessons:
        return False
    existing_text = "\n\n".join(f"Lesson {lesson.id}:\n{lesson.description}" for lesson in existing_lessons)
    prompt = format_prompt(existing_lessons=existing_text, new_lesson=new_lesson.description)
    try:
        result = llm.call_json(prompt)
        is_dup = result.get("duplicate", False)
        if is_dup:
            logger.info("Lesson %s is duplicate: %s", new_lesson.id, result.get("reasoning", ""))
        return is_dup
    except Exception:
        logger.warning("Dedup check failed, assuming not duplicate", exc_info=True)
        return False
