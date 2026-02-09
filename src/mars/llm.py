"""Thin LLM wrapper using langchain's universal model init for multi-provider support."""

from __future__ import annotations

import json
import logging
import os
import re
import time

from langchain.chat_models import init_chat_model

from mars.config import MARSConfig

# Custom provider aliases that remap to native langchain providers.
# Format: alias -> (langchain_provider, base_url, env_var_for_api_key)
_PROVIDER_ALIASES: dict[str, tuple[str, str, str]] = {
    "openrouter": ("openai", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}

logger = logging.getLogger(__name__)


class LLMClient:
    """Multi-provider LLM client with retry logic and structured output parsing.

    Supports any provider via "provider:model" format (e.g. "openai:gpt-4o",
    "anthropic:claude-sonnet-4-20250514", "google_genai:gemini-2.5-pro").
    Auto-infers provider from model name when prefix is omitted.

    Custom aliases (automatically remapped):
      - "openrouter:<model>" -> openai provider with OpenRouter base URL
    """

    def __init__(self, config: MARSConfig) -> None:
        self.config = config
        model_name, extra_kwargs = _resolve_model(config.model_name)
        init_kwargs: dict[str, object] = {"temperature": config.temperature, **extra_kwargs}
        self.model = init_chat_model(model_name, **init_kwargs)  # type: ignore[arg-type]
        self.total_calls = 0

    def call(self, prompt: str, *, system: str | None = None) -> str:
        """Generate text response from LLM."""
        messages: list[tuple[str, str]] = []
        if system:
            messages.append(("system", system))
        messages.append(("human", prompt))
        return self._invoke_with_retry(messages)

    def call_json(self, prompt: str, *, system: str | None = None) -> dict:
        """Generate JSON-structured response from LLM."""
        raw = self.call(prompt, system=system)
        return _extract_json(raw)

    def call_code(self, prompt: str, *, system: str | None = None) -> str:
        """Generate code response, extracting from markdown code blocks."""
        raw = self.call(prompt, system=system)
        return _extract_code(raw)

    def _invoke_with_retry(
        self,
        messages: list[tuple[str, str]],
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> str:
        """Invoke LLM with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                response = self.model.invoke(messages)
                self.total_calls += 1
                content = response.content
                if isinstance(content, str):
                    return content
                # Handle models that return list of content blocks (e.g. thinking + text)
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, str):
                            text_parts.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif isinstance(block, dict) and block.get("type") == "thinking":
                            continue  # Skip thinking blocks
                    return "\n".join(text_parts)
                return str(content)
            except Exception:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2**attempt)
                logger.warning("LLM call failed (attempt %d/%d), retrying in %.1fs", attempt + 1, max_retries, delay)
                time.sleep(delay)
        raise RuntimeError("unreachable")


def _resolve_model(model_name: str) -> tuple[str, dict[str, str]]:
    """Resolve custom provider aliases to langchain-native provider + kwargs.

    For example, "openrouter:meta-llama/llama-3-70b" becomes
    ("openai:meta-llama/llama-3-70b", {"base_url": "https://...", "api_key": "..."}).

    Returns (model_name, extra_kwargs) for init_chat_model.
    """
    if ":" not in model_name:
        return model_name, {}

    prefix, model = model_name.split(":", 1)
    alias = _PROVIDER_ALIASES.get(prefix)
    if alias is None:
        return model_name, {}

    provider, base_url, env_var = alias
    api_key = os.environ.get(env_var, "")
    if not api_key:
        raise ValueError(f"{env_var} environment variable is required for {prefix}: provider")

    logger.info("Remapping %s -> %s:%s (base_url=%s)", model_name, provider, model, base_url)
    return f"{provider}:{model}", {"base_url": base_url, "api_key": api_key}


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response, handling markdown code blocks."""
    # Try to find JSON in code block first
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Handle double-braced JSON (common in LLM outputs)
    if text.startswith("{{") and text.endswith("}}"):
        text = text[1:-1]

    return json.loads(text)  # type: ignore[no-any-return]


def _extract_code(text: str) -> str:
    """Extract code from markdown code blocks."""
    match = re.search(r"```(?:python)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # If no code block, return the entire text (some models skip the fences)
    return text.strip()
