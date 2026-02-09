"""Comprehensive test suite for MARS framework core modules."""

import tempfile
from pathlib import Path

import pytest

from mars.config import MARSConfig
from mars.mcts.reward import compute_reward
from mars.mcts.selection import backpropagate, select_node
from mars.mcts.tree import MCTSNode, MCTSTree
from mars.memory.lesson_pool import Lesson, LessonPool
from mars.prompts import initial_idea, lesson_dedup, metric_parsing
from mars.solution.diff import apply_diffs, parse_diffs

# ============================================================================
# 1. Test config.py - MARSConfig
# ============================================================================


class TestMARSConfig:
    """Test MARSConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MARSConfig()
        assert config.max_lessons == 30
        assert config.max_debug_attempts == 10
        assert config.max_improvements == 2
        assert config.reward_weight == -0.07
        assert config.uct_constant == 1.414
        assert config.stale_threshold == 3
        assert config.num_model_candidates == 5
        assert config.exec_timeout == 86400
        assert config.script_timeout == 14400
        assert config.model_name == "gemini-2.5-pro"
        assert config.temperature == 1.0
        assert config.work_dir == "./workspace"
        assert config.input_dir == "./input"
        assert config.node_counter == 0

    def test_custom_values(self) -> None:
        """Test creating config with custom values."""
        config = MARSConfig(
            max_lessons=50,
            max_debug_attempts=15,
            uct_constant=2.0,
            model_name="gpt-4",
            temperature=0.7,
        )
        assert config.max_lessons == 50
        assert config.max_debug_attempts == 15
        assert config.uct_constant == 2.0
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7

    def test_next_node_id_counter(self) -> None:
        """Test node ID generation increments correctly."""
        config = MARSConfig()
        assert config.next_node_id() == "node_00000"
        assert config.next_node_id() == "node_00001"
        assert config.next_node_id() == "node_00002"
        assert config.node_counter == 3

    def test_next_node_id_custom_start(self) -> None:
        """Test node ID generation with custom starting counter."""
        config = MARSConfig(node_counter=100)
        assert config.next_node_id() == "node_00100"
        assert config.next_node_id() == "node_00101"


# ============================================================================
# 2. Test mcts/tree.py - MCTSNode and MCTSTree
# ============================================================================


class TestMCTSNode:
    """Test MCTSNode class."""

    def test_node_creation(self) -> None:
        """Test basic node creation."""
        node = MCTSNode(id="node_00001")
        assert node.id == "node_00001"
        assert node.parent is None
        assert node.children == []
        assert node.action == "draft"
        assert node.is_buggy is False
        assert node.metric_value is None
        assert node.visit_count == 0
        assert node.q_value == 0.0

    def test_is_valid(self) -> None:
        """Test is_valid() method."""
        # Invalid: buggy node
        node = MCTSNode(id="n1", is_buggy=True, metric_value=0.5, valid_metric=True)
        assert not node.is_valid()

        # Invalid: no metric
        node = MCTSNode(id="n2", is_buggy=False, metric_value=None, valid_metric=True)
        assert not node.is_valid()

        # Invalid: invalid metric flag
        node = MCTSNode(id="n3", is_buggy=False, metric_value=0.5, valid_metric=False)
        assert not node.is_valid()

        # Valid: all conditions met
        node = MCTSNode(id="n4", is_buggy=False, metric_value=0.5, valid_metric=True)
        assert node.is_valid()

    def test_is_root(self) -> None:
        """Test is_root() method."""
        root = MCTSNode(id="root")
        assert root.is_root()

        child = MCTSNode(id="child", parent=root)
        assert not child.is_root()


class TestMCTSTree:
    """Test MCTSTree class."""

    def test_tree_initialization(self) -> None:
        """Test tree initialization."""
        tree = MCTSTree(root_id="root")
        assert tree.root.id == "root"
        assert tree.best_node is None
        assert len(tree.all_nodes) == 1
        assert tree.all_nodes[0] == tree.root

    def test_add_node(self) -> None:
        """Test adding nodes to tree."""
        tree = MCTSTree(root_id="root")
        child1 = MCTSNode(id="child1", parent=tree.root)
        tree.add_node(child1)

        assert len(tree.all_nodes) == 2
        assert child1 in tree.all_nodes
        assert child1 in tree.root.children

        child2 = MCTSNode(id="child2", parent=tree.root)
        tree.add_node(child2)
        assert len(tree.all_nodes) == 3
        assert len(tree.root.children) == 2

    def test_get_valid_nodes(self) -> None:
        """Test filtering valid nodes."""
        tree = MCTSTree(root_id="root")

        # Add valid node
        valid1 = MCTSNode(id="v1", parent=tree.root, is_buggy=False, metric_value=0.8, valid_metric=True)
        tree.add_node(valid1)

        # Add buggy node
        buggy = MCTSNode(id="b1", parent=tree.root, is_buggy=True, metric_value=0.5, valid_metric=True)
        tree.add_node(buggy)

        # Add node with no metric
        no_metric = MCTSNode(id="n1", parent=tree.root, is_buggy=False, metric_value=None, valid_metric=False)
        tree.add_node(no_metric)

        # Add another valid node
        valid2 = MCTSNode(id="v2", parent=tree.root, is_buggy=False, metric_value=0.9, valid_metric=True)
        tree.add_node(valid2)

        valid_nodes = tree.get_valid_nodes()
        assert len(valid_nodes) == 2
        assert valid1 in valid_nodes
        assert valid2 in valid_nodes

    def test_update_best_higher_is_better(self) -> None:
        """Test update_best with higher metric being better."""
        tree = MCTSTree(root_id="root")

        node1 = MCTSNode(id="n1", is_buggy=False, metric_value=0.8, valid_metric=True)
        tree.add_node(node1)

        # First valid node becomes best
        assert tree.update_best(node1, lower_is_better=False)
        assert tree.best_node == node1

        # Lower metric doesn't improve
        node2 = MCTSNode(id="n2", is_buggy=False, metric_value=0.7, valid_metric=True)
        tree.add_node(node2)
        assert not tree.update_best(node2, lower_is_better=False)
        assert tree.best_node == node1

        # Higher metric improves
        node3 = MCTSNode(id="n3", is_buggy=False, metric_value=0.9, valid_metric=True)
        tree.add_node(node3)
        assert tree.update_best(node3, lower_is_better=False)
        assert tree.best_node == node3

    def test_update_best_lower_is_better(self) -> None:
        """Test update_best with lower metric being better."""
        tree = MCTSTree(root_id="root")

        node1 = MCTSNode(id="n1", is_buggy=False, metric_value=0.8, valid_metric=True)
        tree.add_node(node1)

        # First valid node becomes best
        assert tree.update_best(node1, lower_is_better=True)
        assert tree.best_node == node1

        # Higher metric doesn't improve
        node2 = MCTSNode(id="n2", is_buggy=False, metric_value=0.9, valid_metric=True)
        tree.add_node(node2)
        assert not tree.update_best(node2, lower_is_better=True)
        assert tree.best_node == node1

        # Lower metric improves
        node3 = MCTSNode(id="n3", is_buggy=False, metric_value=0.5, valid_metric=True)
        tree.add_node(node3)
        assert tree.update_best(node3, lower_is_better=True)
        assert tree.best_node == node3

    def test_update_best_invalid_nodes(self) -> None:
        """Test that invalid nodes don't become best."""
        tree = MCTSTree(root_id="root")

        buggy = MCTSNode(id="buggy", is_buggy=True, metric_value=0.9, valid_metric=True)
        assert not tree.update_best(buggy, lower_is_better=False)
        assert tree.best_node is None

    def test_render_tree(self) -> None:
        """Test tree visualization rendering."""
        tree = MCTSTree(root_id="root")

        # Add children
        child1 = MCTSNode(id="c1", parent=tree.root, metric_value=0.8, valid_metric=True)
        tree.add_node(child1)

        child2 = MCTSNode(id="c2", parent=tree.root, is_buggy=True)
        tree.add_node(child2)

        # Set best
        tree.best_node = child1

        output = tree.render_tree()
        assert "Solution tree" in output
        assert "root" in output
        assert "c1" in output
        assert "c2" in output
        assert "0.800000" in output
        assert "(best)" in output
        assert "bug" in output


# ============================================================================
# 3. Test mcts/reward.py - compute_reward
# ============================================================================


class TestComputeReward:
    """Test reward computation."""

    def test_compute_reward_single_node(self) -> None:
        """Test reward computation with single valid node."""
        config = MARSConfig()
        node = MCTSNode(id="n1", is_buggy=False, metric_value=0.8, valid_metric=True, execution_time=100.0)
        all_nodes = [node]

        reward = compute_reward(node, all_nodes, config, lower_is_better=False)
        # Single node: G = 0.5 per Eq 3
        assert reward == pytest.approx(0.5 * (100.0 / 14400.0) ** (-0.07), rel=1e-6)

    def test_compute_reward_multiple_nodes_higher_better(self) -> None:
        """Test reward with multiple nodes, higher metric is better."""
        config = MARSConfig()
        n1 = MCTSNode(id="n1", is_buggy=False, metric_value=0.5, valid_metric=True, execution_time=100.0)
        n2 = MCTSNode(id="n2", is_buggy=False, metric_value=0.8, valid_metric=True, execution_time=200.0)
        n3 = MCTSNode(id="n3", is_buggy=False, metric_value=0.9, valid_metric=True, execution_time=150.0)
        all_nodes = [n1, n2, n3]

        # n3 has highest metric (0.9)
        reward = compute_reward(n3, all_nodes, config, lower_is_better=False)
        # G(n3) = (0.9 - 0.5) / (0.9 - 0.5) = 1.0
        g = 1.0
        time_ratio = 150.0 / 14400.0
        expected = g * (time_ratio**-0.07)
        assert reward == pytest.approx(expected, rel=1e-6)

        # n1 has lowest metric (0.5)
        reward = compute_reward(n1, all_nodes, config, lower_is_better=False)
        # G(n1) = (0.5 - 0.5) / (0.9 - 0.5) = 0.0
        g = 0.0
        time_ratio = 100.0 / 14400.0
        expected = g * (time_ratio**-0.07)
        assert reward == pytest.approx(expected, rel=1e-6)

    def test_compute_reward_multiple_nodes_lower_better(self) -> None:
        """Test reward with multiple nodes, lower metric is better."""
        config = MARSConfig()
        n1 = MCTSNode(id="n1", is_buggy=False, metric_value=0.5, valid_metric=True, execution_time=100.0)
        n2 = MCTSNode(id="n2", is_buggy=False, metric_value=0.8, valid_metric=True, execution_time=200.0)
        all_nodes = [n1, n2]

        # n1 has lower metric (better when lower_is_better=True)
        # Metrics negated: -0.5 and -0.8
        # G(n1) = (-0.5 - (-0.8)) / (-0.5 - (-0.8)) = 0.3 / 0.3 = 1.0
        reward = compute_reward(n1, all_nodes, config, lower_is_better=True)
        g = 1.0
        time_ratio = 100.0 / 14400.0
        expected = g * (time_ratio**-0.07)
        assert reward == pytest.approx(expected, rel=1e-6)

    def test_compute_reward_invalid_node(self) -> None:
        """Test that invalid nodes get zero reward."""
        config = MARSConfig()
        buggy = MCTSNode(id="buggy", is_buggy=True)
        valid = MCTSNode(id="valid", is_buggy=False, metric_value=0.8, valid_metric=True)
        all_nodes = [buggy, valid]

        assert compute_reward(buggy, all_nodes, config, lower_is_better=False) == 0.0

    def test_compute_reward_all_same_metric(self) -> None:
        """Test edge case where all nodes have same metric."""
        config = MARSConfig()
        n1 = MCTSNode(id="n1", is_buggy=False, metric_value=0.7, valid_metric=True, execution_time=100.0)
        n2 = MCTSNode(id="n2", is_buggy=False, metric_value=0.7, valid_metric=True, execution_time=200.0)
        all_nodes = [n1, n2]

        # When m_max == m_min, G = 0.5 per Eq 3
        reward = compute_reward(n1, all_nodes, config, lower_is_better=False)
        g = 0.5
        time_ratio = 100.0 / 14400.0
        expected = g * (time_ratio**-0.07)
        assert reward == pytest.approx(expected, rel=1e-6)


# ============================================================================
# 4. Test mcts/selection.py - select_node and backpropagate
# ============================================================================


class TestSelection:
    """Test UCT selection and backpropagation."""

    def test_select_node_unexpanded_root(self) -> None:
        """Test selection returns root when root is not fully expanded."""
        config = MARSConfig()
        tree = MCTSTree(root_id="root")
        tree.root.fully_expanded = False

        selected = select_node(tree, config)
        assert selected == tree.root

    def test_select_node_with_children(self) -> None:
        """Test UCT selection with children."""
        config = MARSConfig()
        tree = MCTSTree(root_id="root")

        # Create children with different Q values
        child1 = MCTSNode(id="c1", parent=tree.root, q_value=0.5, visit_count=10)
        child2 = MCTSNode(id="c2", parent=tree.root, q_value=0.8, visit_count=5)
        child3 = MCTSNode(id="c3", parent=tree.root, q_value=0.3, visit_count=20)

        tree.add_node(child1)
        tree.add_node(child2)
        tree.add_node(child3)

        tree.root.visit_count = 35
        tree.root.fully_expanded = True

        # Mark children as not fully expanded
        child1.fully_expanded = False
        child2.fully_expanded = False
        child3.fully_expanded = False

        selected = select_node(tree, config)
        # Should select one of the unexpanded children
        assert selected in [child1, child2, child3]

    def test_backpropagate(self) -> None:
        """Test backpropagation updates Q values and visit counts."""
        tree = MCTSTree(root_id="root")

        child = MCTSNode(id="c1", parent=tree.root)
        grandchild = MCTSNode(id="gc1", parent=child)

        tree.add_node(child)
        tree.add_node(grandchild)

        # Backpropagate from grandchild
        reward = 0.8
        backpropagate(grandchild, reward)

        # All nodes in path should be updated
        assert grandchild.visit_count == 1
        assert grandchild.q_value == pytest.approx(0.8)

        assert child.visit_count == 1
        assert child.q_value == pytest.approx(0.8)

        assert tree.root.visit_count == 1
        assert tree.root.q_value == pytest.approx(0.8)

    def test_backpropagate_incremental_mean(self) -> None:
        """Test backpropagation uses incremental mean update."""
        node = MCTSNode(id="n1")

        # First backprop
        backpropagate(node, 0.8)
        assert node.visit_count == 1
        assert node.q_value == pytest.approx(0.8)

        # Second backprop
        backpropagate(node, 0.6)
        assert node.visit_count == 2
        # Q = 0.8 + (0.6 - 0.8) / 2 = 0.7
        assert node.q_value == pytest.approx(0.7)

        # Third backprop
        backpropagate(node, 0.9)
        assert node.visit_count == 3
        # Q = 0.7 + (0.9 - 0.7) / 3 = 0.7666...
        assert node.q_value == pytest.approx(0.766666, rel=1e-5)


# ============================================================================
# 5. Test memory/lesson_pool.py - Lesson and LessonPool
# ============================================================================


class TestLesson:
    """Test Lesson dataclass."""

    def test_lesson_creation(self) -> None:
        """Test basic lesson creation."""
        lesson = Lesson(id="L001", category="solution", description="Always normalize inputs")
        assert lesson.id == "L001"
        assert lesson.category == "solution"
        assert lesson.description == "Always normalize inputs"
        assert lesson.source_node == ""

    def test_lesson_with_source(self) -> None:
        """Test lesson with source node."""
        lesson = Lesson(id="L002", category="debug", description="Check tensor shapes", source_node="node_00042")
        assert lesson.source_node == "node_00042"


class TestLessonPool:
    """Test LessonPool class."""

    def test_pool_initialization(self) -> None:
        """Test pool initialization."""
        pool = LessonPool(max_lessons=10, category="solution")
        assert pool.max_lessons == 10
        assert pool.category == "solution"
        assert len(pool.lessons) == 0

    def test_add_lesson(self) -> None:
        """Test adding lessons to pool."""
        pool = LessonPool(max_lessons=3)
        l1 = Lesson(id="L1", category="solution", description="Lesson 1")
        l2 = Lesson(id="L2", category="solution", description="Lesson 2")

        assert pool.add(l1)
        assert len(pool.lessons) == 1

        assert pool.add(l2)
        assert len(pool.lessons) == 2

    def test_eviction_when_full(self) -> None:
        """Test that oldest lesson is evicted when pool is full."""
        pool = LessonPool(max_lessons=2)
        l1 = Lesson(id="L1", category="solution", description="Lesson 1")
        l2 = Lesson(id="L2", category="solution", description="Lesson 2")
        l3 = Lesson(id="L3", category="solution", description="Lesson 3")

        pool.add(l1)
        pool.add(l2)
        assert len(pool.lessons) == 2

        # Adding third lesson should evict first
        pool.add(l3)
        assert len(pool.lessons) == 2
        assert l1 not in pool.lessons
        assert l2 in pool.lessons
        assert l3 in pool.lessons

    def test_format_lessons_empty(self) -> None:
        """Test formatting empty pool."""
        pool = LessonPool()
        assert pool.format_lessons() == "No lessons available."

    def test_format_lessons(self) -> None:
        """Test formatting lessons for LLM context."""
        pool = LessonPool()
        pool.add(Lesson(id="L1", category="solution", description="Normalize data"))
        pool.add(Lesson(id="L2", category="solution", description="Use dropout"))

        formatted = pool.format_lessons()
        assert "Lesson 1 (ID: L1):" in formatted
        assert "Normalize data" in formatted
        assert "Lesson 2 (ID: L2):" in formatted
        assert "Use dropout" in formatted

    def test_save_and_load(self) -> None:
        """Test saving and loading lessons to/from JSON."""
        pool = LessonPool(category="solution")
        pool.add(Lesson(id="L1", category="solution", description="Lesson 1"))
        pool.add(Lesson(id="L2", category="solution", description="Lesson 2"))

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            # Save
            pool.save(temp_path)

            # Load into new pool
            new_pool = LessonPool(category="solution")
            new_pool.load(temp_path)

            assert len(new_pool.lessons) == 2
            assert new_pool.lessons[0].id == "L1"
            assert new_pool.lessons[0].description == "Lesson 1"
            assert new_pool.lessons[1].id == "L2"
            assert new_pool.lessons[1].description == "Lesson 2"
        finally:
            Path(temp_path).unlink()


# ============================================================================
# 6. Test solution/diff.py - parse_diffs and apply_diffs
# ============================================================================


class TestDiffParsing:
    """Test diff parsing and application."""

    def test_parse_single_diff(self) -> None:
        """Test parsing single diff block."""
        response = """
Some preamble text.

[target file: model.py]
<<<< SEARCH
def old_function():
    pass
====
def new_function():
    return 42
>>>> REPLACE

Some trailing text.
"""
        diffs = parse_diffs(response)
        assert len(diffs) == 1
        assert diffs[0]["file"] == "model.py"
        assert diffs[0]["search"] == "def old_function():\n    pass"
        assert diffs[0]["replace"] == "def new_function():\n    return 42"

    def test_parse_multiple_diffs(self) -> None:
        """Test parsing multiple diff blocks."""
        response = """
[target file: file1.py]
<<<< SEARCH
old code 1
====
new code 1
>>>> REPLACE

[target file: file2.py]
<<<< SEARCH
old code 2
====
new code 2
>>>> REPLACE
"""
        diffs = parse_diffs(response)
        assert len(diffs) == 2
        assert diffs[0]["file"] == "file1.py"
        assert diffs[1]["file"] == "file2.py"

    def test_parse_no_diffs(self) -> None:
        """Test parsing response with no diff blocks."""
        response = "Just some regular text without diffs."
        diffs = parse_diffs(response)
        assert len(diffs) == 0

    def test_apply_diff_to_module(self) -> None:
        """Test applying diff to existing module."""
        modules = {"model.py": "def old_func():\n    return 1\n"}
        main_script = "print('hello')"

        response = """
[target file: model.py]
<<<< SEARCH
def old_func():
    return 1
====
def old_func():
    return 2
>>>> REPLACE
"""
        new_modules, new_main = apply_diffs(response, modules, main_script)

        assert new_modules["model.py"] == "def old_func():\n    return 2\n"
        assert new_main == main_script  # Unchanged

    def test_apply_diff_to_main_script(self) -> None:
        """Test applying diff to main script."""
        modules = {"model.py": "code"}
        main_script = "print('old')\nother_line()"

        response = """
[target file: runfile.py]
<<<< SEARCH
print('old')
====
print('new')
>>>> REPLACE
"""
        new_modules, new_main = apply_diffs(response, modules, main_script)

        assert new_main == "print('new')\nother_line()"
        assert new_modules == modules  # Unchanged

    def test_apply_diff_library_prefix(self) -> None:
        """Test applying diff with library/ prefix."""
        modules = {"utils.py": "def helper():\n    pass\n"}
        main_script = "import utils"

        response = """
[target file: library/utils.py]
<<<< SEARCH
def helper():
    pass
====
def helper():
    return 42
>>>> REPLACE
"""
        new_modules, new_main = apply_diffs(response, modules, main_script)

        assert new_modules["utils.py"] == "def helper():\n    return 42\n"

    def test_apply_diff_creates_new_file(self) -> None:
        """Test that diff for non-existent file creates new file."""
        modules = {"existing.py": "existing code"}
        main_script = "print('hello')"

        # When file doesn't exist and search not found, it creates new file with replace content
        response = """
[target file: new_file.py]
<<<< SEARCH
placeholder
====
def new_function():
    return 100
>>>> REPLACE
"""
        new_modules, new_main = apply_diffs(response, modules, main_script)

        assert "new_file.py" in new_modules
        assert new_modules["new_file.py"] == "def new_function():\n    return 100"
        assert "existing.py" in new_modules  # Original unchanged

    def test_apply_diff_search_not_found(self) -> None:
        """Test that diff is skipped when search block not found."""
        modules = {"model.py": "existing code"}
        main_script = "print('hello')"

        response = """
[target file: model.py]
<<<< SEARCH
non-existent code
====
new code
>>>> REPLACE
"""
        new_modules, new_main = apply_diffs(response, modules, main_script)

        # Should be unchanged since search block not found
        assert new_modules["model.py"] == "existing code"


# ============================================================================
# 7. Test prompts/ - Template rendering
# ============================================================================


class TestPromptTemplates:
    """Test prompt template rendering."""

    def test_metric_parsing_prompt(self) -> None:
        """Test metric_parsing prompt renders correctly."""
        prompt = metric_parsing.format_prompt(task_description="Predict house prices using MSE")

        assert "Task" in prompt
        assert "metric_name" in prompt
        assert "lower_is_better" in prompt
        assert "Predict house prices using MSE" in prompt
        assert "JSON format" in prompt

    def test_initial_idea_prompt(self) -> None:
        """Test initial_idea prompt renders correctly."""
        prompt = initial_idea.format_prompt(
            model_arch_desc="CNN, RNN, Transformer",
            previous_ideas="Idea 1: Use ResNet\nIdea 2: Use LSTM",
            context="Additional context here",
        )

        assert "Model Architectures" in prompt
        assert "CNN, RNN, Transformer" in prompt
        assert "Previous Ideas" in prompt
        assert "Use ResNet" in prompt
        assert "Use LSTM" in prompt
        assert "Additional context here" in prompt
        assert "baseline approach" in prompt

    def test_lesson_dedup_prompt(self) -> None:
        """Test lesson_dedup prompt renders correctly."""
        existing = "Lesson 1: Always normalize\nLesson 2: Use regularization"
        new = "Always normalize input features"

        prompt = lesson_dedup.format_prompt(existing_lessons=existing, new_lesson=new)

        assert "Existing Lessons" in prompt
        assert "Always normalize" in prompt
        assert "Use regularization" in prompt
        assert "New Lesson" in prompt
        assert "Always normalize input features" in prompt
        assert "duplicate" in prompt
        assert "JSON" in prompt

    def test_metric_parsing_json_structure(self) -> None:
        """Test that metric_parsing prompt includes expected JSON structure."""
        prompt = metric_parsing.format_prompt(task_description="Test task")

        # Should show example JSON with metric_name and lower_is_better
        assert '"metric_name"' in prompt
        assert '"lower_is_better"' in prompt
        assert '"accuracy"' in prompt  # Example value
        assert "false" in prompt  # Example value for accuracy

    def test_initial_idea_sections(self) -> None:
        """Test that initial_idea prompt includes all required sections."""
        prompt = initial_idea.format_prompt(model_arch_desc="Architectures", previous_ideas="Ideas", context="Context")

        # Should mention required response sections
        assert "Model:" in prompt or "Model" in prompt
        assert "Data:" in prompt or "Data" in prompt
        assert "Training:" in prompt or "Training" in prompt
        assert "Evaluation:" in prompt or "Evaluation" in prompt

    def test_lesson_dedup_guidelines(self) -> None:
        """Test that lesson_dedup prompt includes deduplication guidelines."""
        prompt = lesson_dedup.format_prompt(existing_lessons="Existing", new_lesson="New")

        # Should include semantic overlap guidance
        assert "Semantic" in prompt or "semantic" in prompt
        assert "duplicate" in prompt
        assert "reasoning" in prompt
