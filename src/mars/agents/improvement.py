"""Solution improvement agent for the MARS framework."""

from __future__ import annotations

import logging

from mars.agents.base import BaseAgent
from mars.config import MARSConfig
from mars.llm import LLMClient
from mars.prompts import solution_improvement

logger = logging.getLogger(__name__)


class ImprovementAgent(BaseAgent):
    """Improves an existing solution using accumulated lessons."""

    def __init__(self, llm: LLMClient, config: MARSConfig) -> None:
        super().__init__(llm, config)

    def improve(self, current_solution: str, lessons: str) -> str:
        """Generate improvement edits for the current solution.

        Args:
            current_solution: The current solution code to improve.
            lessons: Accumulated lessons from prior attempts.

        Returns:
            Raw LLM response containing diff-format edits.
        """
        prompt = solution_improvement.format_prompt(
            lessons=lessons,
            previous_solution=current_solution,
        )
        logger.info("Generating solution improvements")
        return self.call(prompt)
