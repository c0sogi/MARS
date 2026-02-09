"""Idea generation agents for the MARS framework."""

from __future__ import annotations

import logging

from mars.agents.base import BaseAgent
from mars.config import MARSConfig
from mars.llm import LLMClient
from mars.prompts import idea_improvement, initial_idea

logger = logging.getLogger(__name__)


class InitialIdeaAgent(BaseAgent):
    """Generates initial solution ideas from model architectures."""

    def __init__(self, llm: LLMClient, config: MARSConfig) -> None:
        super().__init__(llm, config)

    def propose(
        self,
        model_archs: str,
        previous_ideas: list[str],
        context: str,
    ) -> str:
        """Propose an initial idea based on model architectures.

        Args:
            model_archs: Description of candidate model architectures.
            previous_ideas: Previously generated ideas to avoid repetition.
            context: Task/problem context.

        Returns:
            Natural-language idea text.
        """
        prev = "\n".join(previous_ideas) or "None"
        prompt = initial_idea.format_prompt(
            model_arch_desc=model_archs,
            previous_ideas=prev,
            context=context,
        )
        logger.info("Generating initial idea")
        return self.call(prompt)


class IdeaImprovementAgent(BaseAgent):
    """Improves existing ideas using accumulated lessons."""

    def __init__(self, llm: LLMClient, config: MARSConfig) -> None:
        super().__init__(llm, config)

    def propose(
        self,
        previous_ideas: list[str],
        lessons: str,
        context: str,
    ) -> str:
        """Propose an improved idea based on lessons learned.

        Args:
            previous_ideas: Previously generated ideas.
            lessons: Accumulated lessons from prior attempts.
            context: Task/problem context.

        Returns:
            Improved natural-language idea text.
        """
        prompt = idea_improvement.format_prompt(
            previous_ideas="\n".join(previous_ideas),
            lessons=lessons,
            context=context,
        )
        logger.info("Generating improved idea from lessons")
        return self.call(prompt)
