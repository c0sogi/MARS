"""Diff parsing and application for LLM-generated code edits."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def parse_diffs(llm_response: str) -> list[dict]:
    """Parse diff-format edits from LLM response.

    Expected format per edit:
    [target file: filename.py]
    <<<< SEARCH
    old code block
    ====
    new code block
    >>>> REPLACE
    """
    diffs = []
    # Match diff blocks
    pattern = re.compile(
        r"\[target file:\s*(.+?)\]\s*\n"
        r"<<<< SEARCH\s*\n"
        r"(.*?)\n"
        r"====\s*\n"
        r"(.*?)\n"
        r">>>> REPLACE",
        re.DOTALL,
    )
    for match in pattern.finditer(llm_response):
        diffs.append(
            {
                "file": match.group(1).strip(),
                "search": match.group(2).strip(),
                "replace": match.group(3).strip(),
            }
        )
    return diffs


def apply_diffs(
    llm_response: str,
    modules: dict[str, str],
    main_script: str,
) -> tuple[dict[str, str], str]:
    """Apply diff edits from LLM response to modules and main script.

    Returns updated (modules, main_script).
    """
    diffs = parse_diffs(llm_response)
    new_modules = dict(modules)
    new_main = main_script

    for diff in diffs:
        target = diff["file"]
        search = diff["search"]
        replace = diff["replace"]

        if target in ("runfile.py", "main.py"):
            if search in new_main:
                new_main = new_main.replace(search, replace, 1)
                logger.info("Applied diff to %s", target)
            else:
                logger.warning("Search block not found in %s, skipping", target)
        else:
            # Try matching with and without library/ prefix
            fname = target
            if fname.startswith("library/"):
                fname = fname[len("library/") :]

            if fname in new_modules:
                if search in new_modules[fname]:
                    new_modules[fname] = new_modules[fname].replace(search, replace, 1)
                    logger.info("Applied diff to %s", fname)
                else:
                    logger.warning("Search block not found in %s, skipping", fname)
            else:
                # New file
                new_modules[fname] = replace
                logger.info("Created new file %s from diff", fname)

    return new_modules, new_main
