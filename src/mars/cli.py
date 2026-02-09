"""CLI entry point for MARS framework.

Provides subcommands: init, run, list.

Usage:
    mars init my-task --template classification
    mars run my-task --budget 1800
    mars list
"""

from __future__ import annotations

import argparse
import importlib.resources
import logging
import os
import sys

from dotenv import load_dotenv

from mars.config import MARSConfig
from mars.mcts.search import run_mars

TEMPLATES = ("generic", "classification", "regression", "nlp", "vision")
DEFAULT_TASKS_DIR = "tasks"


def _resolve_task_dir(task: str, tasks_dir: str) -> str:
    """Resolve a task name or path to an absolute task directory."""
    # If it looks like an explicit path (contains separator or starts with .), use as-is
    if os.sep in task or task.startswith(".") or os.path.isabs(task):
        return os.path.abspath(task)
    # Otherwise resolve as a name under tasks_dir
    return os.path.abspath(os.path.join(tasks_dir, task))


def _load_template(template_name: str) -> str:
    """Load a built-in template by name."""
    templates_pkg = importlib.resources.files("mars.templates")
    template_file = templates_pkg.joinpath(f"{template_name}.md")
    return template_file.read_text(encoding="utf-8")


# ── mars init ────────────────────────────────────────────────────────────


def cmd_init(args: argparse.Namespace) -> None:
    """Handle ``mars init <name>``."""
    tasks_dir = args.dir
    task_name: str = args.name
    template_name: str = args.template

    task_dir = os.path.join(os.path.abspath(tasks_dir), task_name)

    if os.path.exists(task_dir):
        print(f"Error: Directory already exists: {task_dir}", file=sys.stderr)
        sys.exit(1)

    # Load template
    try:
        template_content = _load_template(template_name)
    except FileNotFoundError:
        print(
            f"Error: Unknown template '{template_name}'. Available: {', '.join(TEMPLATES)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create structure
    os.makedirs(task_dir)
    os.makedirs(os.path.join(task_dir, "input"))

    # Write description.md
    description = template_content.replace("{task_name}", task_name)
    desc_path = os.path.join(task_dir, "description.md")
    with open(desc_path, "w", encoding="utf-8") as f:
        f.write(description)

    # Print next steps
    rel_dir = os.path.relpath(task_dir)
    print(f"Created {rel_dir}/")
    print(f"  {rel_dir}/description.md   <- edit this")
    print(f"  {rel_dir}/input/           <- put data here")
    print()
    print("Next: edit description.md, copy data to input/, then run:")
    print(f"  mars run {task_name} --budget 1800")


# ── mars run ─────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    """Handle ``mars run <task>``."""
    load_dotenv()

    task_dir = _resolve_task_dir(args.task, args.dir)

    # Validate task directory
    if not os.path.isdir(task_dir):
        print(f"Error: Task directory not found: {task_dir}", file=sys.stderr)
        print(f"Hint: Run 'mars init {args.task}' to create it.", file=sys.stderr)
        sys.exit(1)

    # Load task description
    desc_path = os.path.join(task_dir, "description.md")
    if not os.path.exists(desc_path):
        for alt in ("description.txt", "README.md", "task.md"):
            alt_path = os.path.join(task_dir, alt)
            if os.path.exists(alt_path):
                desc_path = alt_path
                break
        else:
            print(f"Error: No description file found in {task_dir}", file=sys.stderr)
            print("Hint: Create a description.md with task details.", file=sys.stderr)
            sys.exit(1)

    with open(desc_path, encoding="utf-8") as f:
        task_description = f.read()

    # Validate input directory
    input_dir = os.path.join(task_dir, "input")
    if not os.path.isdir(input_dir):
        print(f"Warning: Input directory not found: {input_dir}", file=sys.stderr)

    # Set up workspace
    work_dir = os.path.join(task_dir, "workspace")
    os.makedirs(work_dir, exist_ok=True)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "mars.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Build config
    config = MARSConfig(
        exec_timeout=args.budget,
        script_timeout=args.script_timeout,
        model_name=args.model,
        temperature=args.temperature,
        max_lessons=args.max_lessons,
        max_debug_attempts=args.max_debug,
        work_dir=work_dir,
        input_dir=input_dir,
    )

    logging.info("MARS starting with config: %s", config)
    logging.info("Task: %s", task_dir)
    logging.info("Description length: %d chars", len(task_description))

    best_path = run_mars(task_description, config)
    print(f"\nMARS complete. Best solution: {best_path}")


# ── mars list ────────────────────────────────────────────────────────────


def cmd_list(args: argparse.Namespace) -> None:
    """Handle ``mars list``."""
    tasks_dir = os.path.abspath(args.dir)

    if not os.path.isdir(tasks_dir):
        print(f"No tasks directory found at: {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    entries = sorted(os.listdir(tasks_dir))
    task_dirs = [e for e in entries if os.path.isdir(os.path.join(tasks_dir, e))]

    if not task_dirs:
        print("No tasks found. Create one with: mars init <name>")
        return

    # Header
    print(f"{'TASK':<20} {'DESCRIPTION':<13} {'INPUT':<8} {'WORKSPACE':<11} {'BEST METRIC'}")

    for name in task_dirs:
        td = os.path.join(tasks_dir, name)
        has_desc = any(
            os.path.exists(os.path.join(td, f)) for f in ("description.md", "description.txt", "README.md", "task.md")
        )
        has_input = os.path.isdir(os.path.join(td, "input"))
        has_workspace = os.path.isdir(os.path.join(td, "workspace"))

        best_metric = "-"
        if has_workspace:
            best_metric = _read_best_metric(os.path.join(td, "workspace"))

        print(
            f"{name:<20} {'yes' if has_desc else 'no':<13} {'yes' if has_input else 'no':<8} "
            f"{'yes' if has_workspace else 'no':<11} {best_metric}"
        )


def _read_best_metric(workspace_dir: str) -> str:
    """Try to read the best metric from a workspace's tree.txt or best_solution."""
    tree_path = os.path.join(workspace_dir, "tree.txt")
    if not os.path.exists(tree_path):
        return "-"
    try:
        with open(tree_path, encoding="utf-8") as f:
            content = f.read()
        # Look for the best node marker (★)
        for line in content.splitlines():
            if "\u2605" in line:  # ★ best marker
                # Extract metric value: metric=0.85
                for part in line.split("|"):
                    part = part.strip()
                    if part.startswith("metric="):
                        return part.removeprefix("metric=").strip()
    except OSError:
        pass
    return "-"


# ── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mars",
        description="MARS: Modular Agent with Reflective Search for Automated AI Research",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── mars init ──
    p_init = subparsers.add_parser("init", help="Create a new task from a template")
    p_init.add_argument("name", help="Task name (creates tasks/<name>/)")
    p_init.add_argument(
        "--template",
        default="generic",
        choices=TEMPLATES,
        help="Task template (default: generic)",
    )
    p_init.add_argument("--dir", default=DEFAULT_TASKS_DIR, help="Tasks directory (default: tasks)")
    p_init.set_defaults(func=cmd_init)

    # ── mars run ──
    p_run = subparsers.add_parser("run", help="Run MARS on a task")
    p_run.add_argument("task", help="Task name (resolved to tasks/<name>/) or full path")
    p_run.add_argument("--budget", type=int, default=86400, help="Time budget in seconds (default: 86400 = 24h)")
    p_run.add_argument(
        "--model",
        default="google_genai:gemini-2.5-pro",
        help="LLM model (default: google_genai:gemini-2.5-pro)",
    )
    p_run.add_argument("--temperature", type=float, default=1.0, help="LLM temperature (default: 1.0)")
    p_run.add_argument(
        "--script-timeout", type=int, default=14400, help="Per-script timeout in seconds (default: 14400)"
    )
    p_run.add_argument("--max-lessons", type=int, default=30, help="Max lessons per pool (default: 30)")
    p_run.add_argument("--max-debug", type=int, default=10, help="Max debug attempts per node (default: 10)")
    p_run.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    p_run.add_argument("--dir", default=DEFAULT_TASKS_DIR, help="Tasks directory (default: tasks)")
    p_run.set_defaults(func=cmd_run)

    # ── mars list ──
    p_list = subparsers.add_parser("list", help="List available tasks")
    p_list.add_argument("--dir", default=DEFAULT_TASKS_DIR, help="Tasks directory (default: tasks)")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
