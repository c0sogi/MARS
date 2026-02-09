"""Execution result review agent for the MARS framework."""

from __future__ import annotations

import logging

from mars.agents.base import BaseAgent
from mars.config import MARSConfig
from mars.llm import LLMClient
from mars.prompts import execution_review

logger = logging.getLogger(__name__)


class ReviewAgent(BaseAgent):
    """Reviews execution results to extract metrics and summaries."""

    def __init__(self, llm: LLMClient, config: MARSConfig) -> None:
        super().__init__(llm, config)

    def review(
        self,
        library_files: str,
        code: str,
        execution_output: str,
    ) -> dict:
        """Review execution output and extract structured results.

        Args:
            library_files: Concatenated library module source.
            code: The main solution script.
            execution_output: Terminal output from execution.

        Returns:
            Dict with keys: summary (str), metric (float|None),
            valid_metric (bool).
        """
        prompt = execution_review.format_prompt(
            library_files=library_files,
            code=code,
            term_out=execution_output,
        )
        logger.info("Reviewing execution results")
        return self.call_json(prompt)
