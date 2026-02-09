"""Main MCTS loop implementing Algorithm 2 from the MARS paper.

Orchestrates the full search: task preparation, idea generation,
modular coding, execution, debugging, reward computation,
backpropagation, and lesson extraction.
"""

from __future__ import annotations

import logging
import os
import time

from mars.agents.coding import CodingAgent
from mars.agents.debugging import BugAnalysisAgent, DebuggingAgent
from mars.agents.idea import IdeaImprovementAgent, InitialIdeaAgent
from mars.agents.improvement import ImprovementAgent
from mars.agents.modular import ModularAgent
from mars.agents.review import ReviewAgent
from mars.agents.testing import TestingAgent
from mars.config import MARSConfig
from mars.execution.reviewer import review_execution
from mars.execution.runner import ScriptRunner
from mars.llm import LLMClient
from mars.mcts.reward import compute_reward
from mars.mcts.selection import backpropagate, select_node
from mars.mcts.tree import MCTSNode, MCTSTree
from mars.memory.deduplication import is_duplicate_lesson
from mars.memory.distillation import distill_debug_lesson, distill_solution_lesson
from mars.memory.lesson_pool import LessonPool
from mars.prep.eda import run_eda
from mars.prep.metadata import generate_metadata
from mars.prep.metric_parsing import parse_metric
from mars.prep.model_search import search_architectures
from mars.solution.diff import apply_diffs
from mars.solution.repository import SolutionRepo

logger = logging.getLogger(__name__)


def run_mars(task_description: str, config: MARSConfig) -> str:
    """Run the MARS algorithm (Algorithm 2) and return path to best solution."""
    llm = LLMClient(config)
    start_time = time.time()

    # === Task Preparation (lines 3-7 of Algorithm 2) ===
    logger.info("=== Task Preparation ===")

    # Line 3: MetricParsing(I)
    metric_info = parse_metric(llm, task_description)
    metric_name: str = metric_info["metric_name"]
    lower_is_better: bool = metric_info["lower_is_better"]
    logger.info("Metric: %s (lower_is_better=%s)", metric_name, lower_is_better)

    # Line 4: Preprocess(D) - generate metadata
    metadata_context = generate_metadata(llm, task_description, config)

    # Line 5: Analyze(I, D, C_meta) - EDA
    eda_context = run_eda(llm, task_description, metadata_context, config)

    # Line 6: SearchArchitectures(I) - model search
    model_archs = search_architectures(llm, task_description, config)

    # Line 7: C = {C_meta, C_eda, C_model}
    context = _build_context(
        task_description,
        metadata_context,
        eda_context,
        metric_name,
        lower_is_better,
    )

    # === Initialise agents ===
    initial_idea_agent = InitialIdeaAgent(llm, config)
    idea_improvement_agent = IdeaImprovementAgent(llm, config)
    modular_agent = ModularAgent(llm, config)
    coding_agent = CodingAgent(llm, config)
    testing_agent = TestingAgent(llm, config)
    improvement_agent = ImprovementAgent(llm, config)
    bug_analysis_agent = BugAnalysisAgent(llm, config)
    debugging_agent = DebuggingAgent(llm, config)
    review_agent = ReviewAgent(llm, config)
    runner = ScriptRunner(config)
    repo = SolutionRepo(config.work_dir)

    # Lines 8-12: Initialise tree and pools
    tree = MCTSTree(root_id=config.next_node_id())
    solution_lessons = LessonPool(max_lessons=config.max_lessons, category="solution")
    debug_lessons = LessonPool(max_lessons=config.max_lessons, category="debug")
    explored_ideas: list[str] = []
    idea_counter = 0

    # === Main MCTS Loop (lines 13-37) ===
    logger.info("=== Starting MCTS Loop ===")

    while (time.time() - start_time) < config.exec_timeout:
        elapsed = time.time() - start_time
        logger.info(
            "--- MCTS Iteration (elapsed: %.0fs / %ds) ---",
            elapsed,
            config.exec_timeout,
        )

        # Line 14: Select node via UCT
        selected = select_node(tree, config)
        logger.info("Selected node: %s (root=%s)", selected.id, selected.is_root())

        if selected.is_root():
            # === DRAFT phase (lines 15-22) ===
            idea_counter += 1
            logger.info("=== Draft Phase (idea %d) ===", idea_counter)

            # Line 16: Propose idea
            if not solution_lessons.lessons:
                idea = initial_idea_agent.propose(model_archs, explored_ideas, context)
            else:
                idea = idea_improvement_agent.propose(
                    explored_ideas,
                    solution_lessons.format_lessons(),
                    context,
                )
            explored_ideas.append(idea)

            # Save idea
            idea_dir = os.path.join(config.work_dir, f"idea_{idea_counter}")
            os.makedirs(idea_dir, exist_ok=True)
            with open(os.path.join(idea_dir, "idea.txt"), "w", encoding="utf-8") as f:
                f.write(idea)

            # Line 17: Decompose into modules
            module_specs = modular_agent.decompose(idea, context)

            # Line 18: Implement modules
            modules: dict[str, str] = {}
            for mod_name, mod_desc in module_specs.items():
                if mod_name == "main":
                    continue
                code = coding_agent.implement_module(
                    idea=idea,
                    file_name=f"{mod_name}.py",
                    file_description=mod_desc,
                    existing_files=modules,
                    context=context,
                )
                modules[f"{mod_name}.py"] = code

            # Line 19: Debug modules (unit tests)
            test_code = testing_agent.generate_test(modules)
            if test_code:
                modules["test_modules.py"] = test_code

            # Line 20: Implement main script
            main_desc = module_specs.get("main", "Orchestrate the full pipeline.")
            main_script = coding_agent.implement_main(
                idea=idea,
                modules=modules,
                file_description=main_desc,
                context=context,
            )

            # Line 21: Create draft node
            new_node = MCTSNode(
                id=config.next_node_id(),
                parent=tree.root,
                action="draft",
                idea=idea,
                idea_id=idea_counter,
                modules=modules,
                main_script=main_script,
                module_descriptions=module_specs,
            )
            tree.add_node(new_node)
        else:
            # === IMPROVE phase (line 24) ===
            logger.info("=== Improve Phase (from %s) ===", selected.id)
            new_node = _improve_node(selected, improvement_agent, solution_lessons, config, tree)

        # === Debug loop (lines 26-30) ===
        k = 0
        while True:
            # Execute and review (line 31)
            node_dir = repo.create_node_dir(new_node.id, new_node.idea_id)
            repo.write_solution(new_node, node_dir)
            exec_result = runner.execute(node_dir)

            review = review_execution(review_agent, new_node, exec_result, context)
            new_node.execution_log = exec_result.output
            new_node.execution_time = exec_result.duration
            new_node.review_summary = review.get("summary", "")
            new_node.metric_value = review.get("metric")
            new_node.valid_metric = review.get("valid_metric", False)
            new_node.is_buggy = not exec_result.success or not new_node.valid_metric

            if not new_node.is_buggy or k >= config.max_debug_attempts:
                break

            # Line 28: Debug
            logger.info(
                "Debug attempt %d/%d for %s",
                k + 1,
                config.max_debug_attempts,
                new_node.id,
            )
            error_analysis = bug_analysis_agent.analyze(
                files=_format_files(new_node),
                exec_result=exec_result.output,
                debug_lessons=debug_lessons.format_lessons(),
            )
            new_node.error_analysis = error_analysis

            fixed_files = debugging_agent.fix(
                files=_format_files(new_node),
                exec_result=exec_result.output,
                error_analysis=error_analysis,
                debug_lessons=debug_lessons.format_lessons(),
            )

            # Apply fixes
            if fixed_files:
                _apply_debug_fixes(new_node, fixed_files)

            # Extract debug lesson
            debug_lesson = distill_debug_lesson(
                llm=llm,
                source_files=_format_files(new_node),
                source_exec_result=exec_result.output,
                error_analysis=error_analysis,
                diff="[debug fixes applied]",
                final_exec_result="[pending re-execution]",
            )
            if debug_lesson and not is_duplicate_lesson(llm, debug_lesson, debug_lessons.lessons):
                debug_lessons.add(debug_lesson)

            new_node.debug_count += 1

            # Create debug child node
            debug_node = MCTSNode(
                id=config.next_node_id(),
                parent=new_node,
                action="debug",
                idea=new_node.idea,
                idea_id=new_node.idea_id,
                modules=dict(new_node.modules),
                main_script=new_node.main_script,
                module_descriptions=new_node.module_descriptions,
            )
            tree.add_node(debug_node)
            new_node = debug_node
            k += 1

        # Line 31: Compute reward
        reward = compute_reward(new_node, tree.all_nodes, config, lower_is_better)
        new_node.reward = reward

        # Line 32: Extract solution lesson
        if new_node.is_valid():
            sol_lesson = distill_solution_lesson(
                llm=llm,
                best_solution=(_format_files(tree.best_node) if tree.best_node else ""),
                new_solution=_format_files(new_node),
            )
            if sol_lesson and not is_duplicate_lesson(llm, sol_lesson, solution_lessons.lessons):
                solution_lessons.add(sol_lesson)

        # Line 33: Backpropagate
        backpropagate(new_node, reward)

        # Lines 34-36: Update best
        updated = tree.update_best(new_node, lower_is_better)
        if updated:
            logger.info(
                "New best node: %s (metric=%.6f)",
                new_node.id,
                new_node.metric_value or 0,
            )
            repo.save_best(new_node)

        # Save tree visualisation
        tree_path = os.path.join(config.work_dir, "tree.txt")
        with open(tree_path, "w", encoding="utf-8") as f:
            f.write(tree.render_tree())

        # Save lessons
        lessons_dir = os.path.join(config.work_dir, "saved_lessons")
        os.makedirs(lessons_dir, exist_ok=True)
        solution_lessons.save(os.path.join(lessons_dir, "solution_lesson.json"))
        debug_lessons.save(os.path.join(lessons_dir, "node_debug_lesson.json"))

        logger.info(
            "Tree has %d nodes, %d valid",
            len(tree.all_nodes),
            len(tree.get_valid_nodes()),
        )

    # Line 38-39: Return best solution
    if tree.best_node:
        best_path = repo.save_best(tree.best_node)
        logger.info("MARS complete. Best solution at: %s", best_path)
        return best_path

    logger.warning("MARS complete. No valid solution found.")
    return config.work_dir


def _build_context(
    task_description: str,
    metadata_context: str,
    eda_context: str,
    metric_name: str,
    lower_is_better: bool,
) -> str:
    """Build the combined context string C = {C_meta, C_eda, C_model}."""
    direction = "minimize" if lower_is_better else "maximize"
    return (
        f"==== Task Description ====\n{task_description}\n\n"
        f"==== Metric ====\n{metric_name} ({direction})\n\n"
        f"==== Metadata ====\n{metadata_context}\n\n"
        f"==== EDA Report ====\n{eda_context}\n"
    )


def _improve_node(
    parent: MCTSNode,
    improvement_agent: ImprovementAgent,
    solution_lessons: LessonPool,
    config: MARSConfig,
    tree: MCTSTree,
) -> MCTSNode:
    """Create an improvement child node from *parent*."""
    diffs = improvement_agent.improve(
        current_solution=_format_files(parent),
        lessons=solution_lessons.format_lessons(),
    )

    new_modules = dict(parent.modules)
    new_main = parent.main_script

    if diffs:
        new_modules, new_main = apply_diffs(diffs, new_modules, new_main)

    node = MCTSNode(
        id=config.next_node_id(),
        parent=parent,
        action="improve",
        idea=parent.idea,
        idea_id=parent.idea_id,
        modules=new_modules,
        main_script=new_main,
        module_descriptions=parent.module_descriptions,
    )
    tree.add_node(node)
    return node


def _format_files(node: MCTSNode | None) -> str:
    """Format a node's files as a string for LLM context."""
    if node is None:
        return ""
    parts: list[str] = []
    for fname, code in node.modules.items():
        parts.append(f"==== {fname} ====\n{code}")
    if node.main_script:
        parts.append(f"==== runfile.py ====\n{node.main_script}")
    return "\n\n".join(parts)


def _apply_debug_fixes(node: MCTSNode, fixed_files: dict[str, str]) -> None:
    """Apply debug fixes to node's files."""
    for fname, code in fixed_files.items():
        if fname == "runfile.py":
            node.main_script = code
        else:
            node.modules[fname] = code
