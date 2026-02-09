"""Exploratory data analysis runner."""

from __future__ import annotations

import logging
import os

from mars.config import MARSConfig
from mars.execution.runner import ScriptRunner
from mars.llm import LLMClient
from mars.prompts.eda import format_prompt

logger = logging.getLogger(__name__)


def run_eda(llm: LLMClient, task_description: str, metadata_context: str, config: MARSConfig) -> str:
    """Run EDA and return analysis report."""
    prompt = format_prompt(task_description=task_description, metadata_context=metadata_context)
    script = llm.call_code(prompt)

    eda_dir = os.path.join(config.work_dir, "eda")
    os.makedirs(eda_dir, exist_ok=True)
    script_path = os.path.join(eda_dir, "eda.py")
    with open(script_path, "w") as f:
        f.write(script)

    runner = ScriptRunner(config)
    result = runner.execute(eda_dir, script="eda.py", timeout=600)

    logger.info("EDA complete")
    return result.output
