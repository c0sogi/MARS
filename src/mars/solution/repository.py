"""Solution repository for managing node solution files on disk."""

from __future__ import annotations

import logging
import os
import shutil

from mars.mcts.tree import MCTSNode

logger = logging.getLogger(__name__)


class SolutionRepo:
    """Manages solution file storage and retrieval."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def create_node_dir(self, node_id: str, idea_id: int) -> str:
        """Create directory for a node's solution."""
        node_dir = os.path.join(self.base_dir, f"idea_{idea_id}", node_id)
        os.makedirs(node_dir, exist_ok=True)
        # Create symlinks/copies of input and metadata dirs
        for subdir in ["input", "metadata"]:
            src = os.path.join(self.base_dir, "..", subdir)
            dst = os.path.join(node_dir, subdir)
            if os.path.exists(src) and not os.path.exists(dst):
                # Create symlink (or copy on Windows)
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:
                    pass  # symlinks may fail on Windows without privileges
        return node_dir

    def write_solution(self, node: MCTSNode, path: str) -> None:
        """Write a node's solution files to disk."""
        os.makedirs(path, exist_ok=True)
        # Create library/ subdirectory for modules
        lib_dir = os.path.join(path, "library")
        os.makedirs(lib_dir, exist_ok=True)

        for fname, code in node.modules.items():
            fpath = os.path.join(lib_dir, fname)
            with open(fpath, "w") as f:
                f.write(code)

        # Write main script as runfile.py
        main_path = os.path.join(path, "runfile.py")
        with open(main_path, "w") as f:
            f.write(node.main_script)

        # Write __init__.py for library module
        init_path = os.path.join(lib_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("")

    def save_best(self, node: MCTSNode) -> str:
        """Save the best solution to best_solution/ directory."""
        best_dir = os.path.join(self.base_dir, "best_solution")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        os.makedirs(best_dir, exist_ok=True)
        self.write_solution(node, best_dir)

        # Also save main.py as a copy at top level for MLE-bench compatibility
        main_src = os.path.join(best_dir, "runfile.py")
        main_dst = os.path.join(best_dir, "main.py")
        if os.path.exists(main_src):
            shutil.copy2(main_src, main_dst)

        logger.info("Saved best solution to %s", best_dir)
        return best_dir
