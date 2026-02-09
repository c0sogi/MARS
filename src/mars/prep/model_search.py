"""SOTA model architecture search."""

from __future__ import annotations

import logging

from mars.config import MARSConfig
from mars.llm import LLMClient
from mars.prompts.model_search import format_prompt

logger = logging.getLogger(__name__)


def search_architectures(llm: LLMClient, task_description: str, config: MARSConfig) -> str:
    """Search for SOTA model architectures. Returns formatted architectures string."""
    prompt = format_prompt(task_description=task_description, num_model_candidates=config.num_model_candidates)
    result = llm.call(prompt)
    logger.info("Model architecture search complete")
    return result
