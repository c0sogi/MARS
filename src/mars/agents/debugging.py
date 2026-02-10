"""Bug analysis and debugging agents for the MARS framework."""

from __future__ import annotations

import logging

from mars.agents.base import BaseAgent
from mars.config import MARSConfig
from mars.llm import LLMClient
from mars.prompts import bug_analysis, debugging
from mars.solution.diff import parse_diffs

logger = logging.getLogger(__name__)


class BugAnalysisAgent(BaseAgent):
    """Analyzes execution errors to identify root causes."""

    def __init__(self, llm: LLMClient, config: MARSConfig) -> None:
        super().__init__(llm, config)

    def analyze(
        self,
        files: str,
        exec_result: str,
        debug_lessons: str,
    ) -> str:
        """Analyze execution output to identify bugs.

        Args:
            files: Concatenated source files.
            exec_result: Execution output / error traceback.
            debug_lessons: Lessons from previous debugging attempts.

        Returns:
            Natural-language error analysis.
        """
        prompt = bug_analysis.format_prompt(
            lessons=debug_lessons,
            files=files,
            exec_result=exec_result,
        )
        logger.info("Analyzing execution errors")
        return self.call(prompt)


class DebuggingAgent(BaseAgent):
    """Generates fixes for identified bugs using diff-format edits."""

    def __init__(self, llm: LLMClient, config: MARSConfig) -> None:
        super().__init__(llm, config)

    def fix(
        self,
        files: str,
        exec_result: str,
        error_analysis: str,
        debug_lessons: str,
    ) -> str:
        """Generate fixes for identified bugs as a raw diff response.

        Args:
            files: Concatenated source files.
            exec_result: Execution output / error traceback.
            error_analysis: Analysis from BugAnalysisAgent.
            debug_lessons: Lessons from previous debugging attempts.

        Returns:
            Raw LLM response containing diffs. Empty string if no response.
        """
        prompt = debugging.format_prompt(
            lessons=debug_lessons,
            files=files,
            exec_result=exec_result,
            error_analysis=error_analysis,
        )
        logger.info("Generating bug fixes")
        response = self.call(prompt)

        diffs = parse_diffs(response)
        if diffs:
            logger.info("Parsed %d diffs from debugging response", len(diffs))
            return response
        else:
            logger.warning("No valid diffs found in debugging response")
            return ""
