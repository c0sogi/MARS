"""Node and Tree data structures for MCTS-based solution search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class MCTSNode:
    """A single node in the MCTS search tree.

    Each node represents a solution state (draft, improvement, or debug
    variant) with its associated code modules, execution results, and
    MCTS statistics.
    """

    id: str  # e.g. "node_00001"
    parent: MCTSNode | None = None
    children: list[MCTSNode] = field(default_factory=list)
    action: Literal["draft", "improve", "debug"] = "draft"

    # Solution state
    idea: str = ""  # Natural language idea
    idea_id: int = 0  # Which idea number this belongs to
    modules: dict[str, str] = field(default_factory=dict)  # {filename: code}
    main_script: str = ""  # runfile.py content
    module_descriptions: dict[str, str] = field(default_factory=dict)  # from decomposition

    # Execution results
    is_buggy: bool = False
    metric_value: float | None = None
    execution_time: float | None = None
    execution_log: str = ""
    valid_metric: bool = False
    review_summary: str = ""

    # Debug state
    error_analysis: str = ""
    debug_count: int = 0

    # MCTS statistics
    visit_count: int = 0
    q_value: float = 0.0
    reward: float = 0.0
    fully_expanded: bool = False

    def is_valid(self) -> bool:
        """Node has a valid (non-buggy) execution with a valid metric."""
        return not self.is_buggy and self.valid_metric and self.metric_value is not None

    def is_root(self) -> bool:
        """Return True if this node has no parent (tree root)."""
        return self.parent is None


class MCTSTree:
    """The full MCTS search tree tracking all explored solutions.

    Maintains the root node, all-nodes list, best-known solution, and
    a staleness counter for triggering re-drafts.
    """

    def __init__(self, root_id: str) -> None:
        self.root = MCTSNode(id=root_id, fully_expanded=False)
        self.best_node: MCTSNode | None = None
        self.all_nodes: list[MCTSNode] = [self.root]
        self.explored_ideas: list[str] = []
        self._valid_nodes_since_best: int = 0  # track stale best

    def add_node(self, node: MCTSNode) -> None:
        """Add a node to the tree and link it to its parent."""
        self.all_nodes.append(node)
        if node.parent is not None:
            node.parent.children.append(node)

    def get_valid_nodes(self) -> list[MCTSNode]:
        """Return all nodes with valid, non-buggy executions."""
        return [n for n in self.all_nodes if n.is_valid()]

    def update_best(self, node: MCTSNode, lower_is_better: bool) -> bool:
        """Update best node if *node* is better. Returns True if updated."""
        if not node.is_valid():
            return False
        if self.best_node is None:
            self.best_node = node
            self._valid_nodes_since_best = 0
            return True
        assert self.best_node.metric_value is not None
        assert node.metric_value is not None
        if lower_is_better:
            improved = node.metric_value < self.best_node.metric_value
        else:
            improved = node.metric_value > self.best_node.metric_value
        if improved:
            self.best_node = node
            self._valid_nodes_since_best = 0
            return True
        self._valid_nodes_since_best += 1
        return False

    def render_tree(self) -> str:
        """Render tree.txt visualization matching exp-logs format."""
        lines = ["Solution tree"]
        self._render_node(self.root, lines, indent=0)
        return "\n".join(lines) + "\n"

    def _render_node(self, node: MCTSNode, lines: list[str], indent: int) -> None:
        prefix = "    " * indent
        if node.is_root():
            label = f"\u25cf -1.000000 (ID: {node.id})"
        elif node.is_buggy:
            label = f"\u25cd bug (ID: {node.id})"
        else:
            metric = f"{node.metric_value:.6f}" if node.metric_value is not None else "?.??????"
            best_tag = " (best)" if node is self.best_node else ""
            label = f"\u25cf {metric}{best_tag} (ID: {node.id})"
        lines.append(f"{prefix}{label}")
        for child in node.children:
            self._render_node(child, lines, indent + 1)
