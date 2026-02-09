"""Code generation agents for the MARS framework."""

from __future__ import annotations

import logging

from mars.agents.base import BaseAgent
from mars.config import MARSConfig
from mars.llm import LLMClient
from mars.prompts import module_implementation, solution_drafting

logger = logging.getLogger(__name__)


def _format_library_files(files: dict[str, str]) -> str:
    """Format a dict of {filename: code} into a readable string.

    Each file is presented as:
        ==== filename ====
        <code>
    """
    if not files:
        return ""
    parts: list[str] = []
    for filename, code in files.items():
        parts.append(f"==== {filename} ====\n{code}")
    return "\n".join(parts)


class CodingAgent(BaseAgent):
    """Generates code for individual modules and the main solution script."""

    def __init__(self, llm: LLMClient, config: MARSConfig) -> None:
        super().__init__(llm, config)

    def implement_module(
        self,
        idea: str,
        file_name: str,
        file_description: str,
        existing_files: dict[str, str],
        context: str,
    ) -> str:
        """Generate code for a single library module.

        Args:
            idea: The solution idea driving the implementation.
            file_name: Name of the module file to implement.
            file_description: Description of what this module should do.
            existing_files: Already-implemented modules {filename: code}.
            context: Task/problem context.

        Returns:
            Generated Python code for the module.
        """
        library_files_str = _format_library_files(existing_files)
        prompt = module_implementation.format_prompt(
            idea=idea,
            library_files=library_files_str,
            file_name=file_name,
            file_description=file_description,
            context=context,
        )
        logger.info("Implementing module: %s", file_name)
        return self.call_code(prompt)

    def implement_main(
        self,
        idea: str,
        modules: dict[str, str],
        file_description: str,
        context: str,
    ) -> str:
        """Generate the main solution script that orchestrates all modules.

        Args:
            idea: The solution idea driving the implementation.
            modules: All library modules {filename: code}.
            file_description: Description of the main script's purpose.
            context: Task/problem context.

        Returns:
            Generated Python code for the main script.
        """
        library_files_str = _format_library_files(modules)
        prompt = solution_drafting.format_prompt(
            idea=idea,
            library_files=library_files_str,
            file_description=file_description,
            context=context,
        )
        logger.info("Implementing main solution script")
        return self.call_code(prompt)
