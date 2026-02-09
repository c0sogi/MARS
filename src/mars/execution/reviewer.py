"""Execution result reviewer with pattern matching and LLM fallback."""

from __future__ import annotations

import logging
import re

from mars.agents.review import ReviewAgent
from mars.execution.runner import ExecutionResult
from mars.mcts.tree import MCTSNode

logger = logging.getLogger(__name__)


def review_execution(review_agent: ReviewAgent, node: MCTSNode, exec_result: ExecutionResult, context: str) -> dict:
    """Review execution results using LLM and pattern matching."""
    # Try pattern matching first for "Final Validation Metric: <value>"
    metric = _extract_metric_pattern(exec_result.output)

    if metric is not None and exec_result.success:
        return {"summary": "Metric extracted via pattern matching", "metric": metric, "valid_metric": True}

    # Fall back to LLM-based review
    library_files = ""
    for fname, code in node.modules.items():
        library_files += f"==== {fname} ====\n{code}\n\n"

    code = node.main_script

    try:
        result = review_agent.review(library_files=library_files, code=code, execution_output=exec_result.output)
        return result
    except Exception:
        logger.warning("LLM review failed, returning no metric", exc_info=True)
        return {"summary": "Review failed", "metric": None, "valid_metric": False}


def _extract_metric_pattern(output: str) -> float | None:
    """Extract metric value from 'Final Validation Metric: <value>' pattern."""
    match = re.search(r"Final Validation Metric:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", output)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None
