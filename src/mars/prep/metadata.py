"""Metadata generation for train/val/test splits."""

from __future__ import annotations

import logging
import os

from mars.config import MARSConfig
from mars.execution.runner import ScriptRunner
from mars.llm import LLMClient
from mars.prompts.metadata_documentation import format_prompt as format_doc_prompt
from mars.prompts.metadata_generation import format_prompt
from mars.prompts.validation_verification import format_prompt as format_verification_prompt

logger = logging.getLogger(__name__)


def generate_metadata(llm: LLMClient, task_description: str, config: MARSConfig) -> str:
    """Generate train/val/test metadata files. Returns metadata documentation string."""
    # Generate metadata script
    prompt = format_prompt(task_description=task_description, exec_timeout=config.script_timeout)
    script = llm.call_code(prompt)

    # Execute metadata generation
    metadata_dir = os.path.join(config.work_dir, "metadata_generation")
    os.makedirs(metadata_dir, exist_ok=True)
    script_path = os.path.join(metadata_dir, "generate_metadata.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)

    runner = ScriptRunner(config)
    result = runner.execute(metadata_dir, script="generate_metadata.py", timeout=600)

    # Verify validation dataset
    verify_prompt = format_verification_prompt(code=script, term_out=result.output)
    verification = llm.call_json(verify_prompt)

    if not verification.get("success", False):
        logger.warning("Metadata verification failed: %s", verification.get("analysis", "unknown"))

    # Document metadata
    doc_prompt = format_doc_prompt(code=script, term_out=result.output)
    documentation = llm.call(doc_prompt)

    # Save documentation
    doc_path = os.path.join(metadata_dir, "metadata_info.txt")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(documentation)

    logger.info("Metadata generation complete")
    return documentation
