"""Script execution runner with timeout and output capture."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from mars.config import MARSConfig

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a script execution."""

    success: bool
    output: str
    duration: float
    return_code: int


class ScriptRunner:
    """Runs Python scripts in isolated directories with timeout support."""

    def __init__(self, config: MARSConfig) -> None:
        self.config = config

    @staticmethod
    def _ensure_input_link(work_dir: str, input_dir: str) -> None:
        """Ensure ./input exists in work_dir, pointing to the actual input directory."""
        dst = os.path.join(work_dir, "input")
        if os.path.exists(dst) or os.path.islink(dst):
            return
        src = os.path.abspath(input_dir)
        if not os.path.isdir(src):
            return
        # Try symlink first, then Windows junction, then skip
        try:
            os.symlink(src, dst, target_is_directory=True)
            return
        except OSError:
            pass
        try:
            # Windows directory junction (works without admin)
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", dst, src],
                capture_output=True, timeout=10,
            )
            if os.path.exists(dst):
                return
        except Exception:
            pass
        logger.warning("Could not create input link in %s", work_dir)

    def execute(self, work_dir: str, *, script: str = "runfile.py", timeout: int | None = None) -> ExecutionResult:
        """Execute a Python script in work_dir with timeout."""
        if timeout is None:
            timeout = self.config.script_timeout

        script_path = os.path.join(work_dir, script)
        if not os.path.exists(script_path):
            return ExecutionResult(
                success=False, output=f"Script not found: {script_path}", duration=0.0, return_code=-1
            )

        # Ensure ./input is accessible from the working directory
        self._ensure_input_link(work_dir, self.config.input_dir)

        logger.info("Executing %s in %s (timeout=%ds)", script, work_dir, timeout)
        start = time.time()

        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - start
            output = result.stdout
            if result.stderr:
                output += "\n=== STDERR ===\n" + result.stderr

            # Truncate very long output
            if len(output) > 100_000:
                output = output[:50_000] + "\n...[truncated]...\n" + output[-50_000:]

            success = result.returncode == 0
            logger.info(
                "Execution %s in %.1fs (rc=%d)", "succeeded" if success else "failed", duration, result.returncode
            )
            return ExecutionResult(success=success, output=output, duration=duration, return_code=result.returncode)
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            logger.warning("Execution timed out after %.1fs", duration)
            return ExecutionResult(
                success=False, output=f"Execution timed out after {timeout}s", duration=duration, return_code=-1
            )
        except Exception as e:
            duration = time.time() - start
            logger.error("Execution error: %s", e)
            return ExecutionResult(success=False, output=str(e), duration=duration, return_code=-1)
