"""CLI entry point for MARS framework.

Usage: uv run python -m mars.run --task ./tasks/my-competition --budget 86400
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from mars.config import MARSConfig
from mars.mcts.search import run_mars


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="MARS: Modular Agent with Reflective Search for Automated AI Research",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Path to task directory containing description.md and input/",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=86400,
        help="Wall-clock time budget in seconds (default: 86400 = 24h)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google_genai:gemini-2.5-pro",
        help="LLM model name, e.g. google_genai:gemini-2.5-pro, openai:gpt-4o, anthropic:claude-sonnet-4-20250514",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="LLM temperature (default: 1.0)",
    )
    parser.add_argument(
        "--script-timeout",
        type=int,
        default=14400,
        help="Per-script execution timeout in seconds (default: 14400 = 4h)",
    )
    parser.add_argument(
        "--max-lessons",
        type=int,
        default=30,
        help="Maximum lessons per pool (default: 30)",
    )
    parser.add_argument(
        "--max-debug",
        type=int,
        default=10,
        help="Maximum debug attempts per node (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Validate task directory
    task_dir = os.path.abspath(args.task)
    if not os.path.isdir(task_dir):
        print(f"Error: Task directory not found: {task_dir}", file=sys.stderr)
        sys.exit(1)

    # Load task description
    desc_path = os.path.join(task_dir, "description.md")
    if not os.path.exists(desc_path):
        # Try alternative names
        for alt in ["description.txt", "README.md", "task.md"]:
            alt_path = os.path.join(task_dir, alt)
            if os.path.exists(alt_path):
                desc_path = alt_path
                break
        else:
            print(f"Error: No description file found in {task_dir}", file=sys.stderr)
            sys.exit(1)

    with open(desc_path) as f:
        task_description = f.read()

    # Set up workspace
    work_dir = os.path.join(task_dir, "workspace")
    os.makedirs(work_dir, exist_ok=True)

    input_dir = os.path.join(task_dir, "input")
    if not os.path.isdir(input_dir):
        print(f"Warning: Input directory not found: {input_dir}", file=sys.stderr)

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

    # Add file logging
    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "mars.log"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logging.info("MARS starting with config: %s", config)
    logging.info("Task: %s", task_dir)
    logging.info("Description length: %d chars", len(task_description))

    # Run MARS
    best_path = run_mars(task_description, config)
    print(f"\nMARS complete. Best solution: {best_path}")


def __main__() -> None:
    main()


if __name__ == "__main__":
    main()
