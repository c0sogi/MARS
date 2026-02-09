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
    ) -> dict[str, str]:
        """Generate and apply fixes for identified bugs.

        Args:
            files: Concatenated source files.
            exec_result: Execution output / error traceback.
            error_analysis: Analysis from BugAnalysisAgent.
            debug_lessons: Lessons from previous debugging attempts.

        Returns:
            Dict of {filename: fixed_code}. Empty dict if parsing fails.
        """
        prompt = debugging.format_prompt(
            lessons=debug_lessons,
            files=files,
            exec_result=exec_result,
            error_analysis=error_analysis,
        )
        logger.info("Generating bug fixes")
        response = self.call(prompt)

        try:
            diffs = parse_diffs(response)
            logger.info("Parsed %d diffs from debugging response", len(diffs))
            # Convert list of diffs to {filename: fixed_code} by collecting replacements
            result: dict[str, str] = {}
            for diff in diffs:
                fname = diff.get("file", "")
                replace = diff.get("replace", "")
                if fname and replace:
                    result[fname] = replace
            return result
        except Exception:
            logger.warning("Failed to parse diffs from debugging response")
            return {}
