"""Module decomposition agent for the MARS framework."""

from __future__ import annotations

import logging

from mars.agents.base import BaseAgent
from mars.config import MARSConfig
from mars.llm import LLMClient
from mars.prompts import modular_decomposition

logger = logging.getLogger(__name__)


class ModularAgent(BaseAgent):
    """Decomposes a solution idea into independent modules."""

    def __init__(self, llm: LLMClient, config: MARSConfig) -> None:
        super().__init__(llm, config)

    def decompose(self, idea: str, context: str) -> dict[str, str]:
        """Decompose an idea into a mapping of module names to descriptions.

        Args:
            idea: The solution idea to decompose.
            context: Task/problem context.

        Returns:
            Dict mapping module filename to its description.
        """
        prompt = modular_decomposition.format_prompt(
            idea=idea,
            context=context,
        )
        logger.info("Decomposing idea into modules")
        return self.call_json(prompt)
