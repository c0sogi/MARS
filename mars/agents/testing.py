"""Module testing agent for the MARS framework."""

from __future__ import annotations

import logging

from mars.agents.base import BaseAgent
from mars.agents.coding import _format_library_files
from mars.config import MARSConfig
from mars.llm import LLMClient
from mars.prompts import module_testing

logger = logging.getLogger(__name__)


class TestingAgent(BaseAgent):
    """Generates test scripts for library modules."""

    def __init__(self, llm: LLMClient, config: MARSConfig) -> None:
        super().__init__(llm, config)

    def generate_test(self, modules: dict[str, str]) -> str:
        """Generate a test script that validates all library modules.

        Args:
            modules: Library modules {filename: code} to test.

        Returns:
            Generated Python test code.
        """
        prompt = module_testing.format_prompt(
            library_files=_format_library_files(modules),
        )
        logger.info("Generating module tests")
        return self.call_code(prompt)
