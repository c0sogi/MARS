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
    Reverts individual files if diff application produces invalid syntax.
    """
    import ast

    diffs = parse_diffs(llm_response)
    new_modules = dict(modules)
    new_main = main_script

    for diff in diffs:
        target = diff["file"]
        search = diff["search"]
        replace = diff["replace"]

        if target in ("runfile.py", "main.py"):
            if search in new_main:
                candidate = new_main.replace(search, replace, 1)
                # Validate syntax after diff application
                try:
                    ast.parse(candidate, filename=target)
                    new_main = candidate
                    logger.info("Applied diff to %s", target)
                except SyntaxError as e:
                    logger.warning(
                        "Diff to %s produces syntax error (%s:%s), reverting",
                        target, e.lineno, e.msg,
                    )
            else:
                logger.warning("Search block not found in %s, skipping", target)
        else:
            # Try matching with and without library/ prefix
            fname = target
            if fname.startswith("library/"):
                fname = fname[len("library/"):]

            if fname in new_modules:
                if search in new_modules[fname]:
                    candidate = new_modules[fname].replace(search, replace, 1)
                    # Validate syntax after diff application
                    try:
                        ast.parse(candidate, filename=fname)
                        new_modules[fname] = candidate
                        logger.info("Applied diff to %s", fname)
                    except SyntaxError as e:
                        logger.warning(
                            "Diff to %s produces syntax error (%s:%s), reverting",
                            fname, e.lineno, e.msg,
                        )
                else:
                    logger.warning("Search block not found in %s, skipping", fname)
            else:
                # New file - validate before adding
                try:
                    ast.parse(replace, filename=fname)
                    new_modules[fname] = replace
                    logger.info("Created new file %s from diff", fname)
                except SyntaxError as e:
                    logger.warning(
                        "New file %s has syntax error (%s:%s), skipping",
                        fname, e.lineno, e.msg,
                    )

    return new_modules, new_main
