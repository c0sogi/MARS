"""Base agent class for all MARS agents."""

from __future__ import annotations

import logging

from mars.config import MARSConfig
from mars.llm import LLMClient

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all MARS agents."""

    def __init__(self, llm: LLMClient, config: MARSConfig) -> None:
        self.llm = llm
        self.config = config

    def call(self, prompt: str, *, system: str | None = None) -> str:
        """Generate a plain-text response from the LLM."""
        return self.llm.call(prompt, system=system)

    def call_json(self, prompt: str, *, system: str | None = None) -> dict:
        """Generate a JSON-structured response from the LLM."""
        return self.llm.call_json(prompt, system=system)

    def call_code(self, prompt: str, *, system: str | None = None) -> str:
        """Generate a code response from the LLM."""
        return self.llm.call_code(prompt, system=system)
