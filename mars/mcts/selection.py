"""UCT-based node selection and backpropagation for the MCTS tree.

Selection rules from Section 4.4.2 of the MARS paper:

* UCT formula (Eq 6): UCT(v) = Q(v) + c_uct * sqrt(ln(N_parent) / N(v))
* Start at root, traverse selecting max-UCT child.
* Stop at node that is not fully expanded.
* Root is fully expanded UNLESS: (1) it has no children, OR
  (2) best hasn't improved in n_s valid nodes.
* If traversal reaches a fully-expanded leaf, re-activate root.
* Buggy nodes are ALWAYS fully expanded.
* Valid nodes are fully expanded when they have >= N_i improvement children.
"""

from __future__ import annotations

import math

from mars.config import MARSConfig
from mars.mcts.tree import MCTSNode, MCTSTree


def select_node(tree: MCTSTree, config: MARSConfig) -> MCTSNode:
    """Select a node for expansion using UCT. Returns the selected node."""
    _update_expansion_status(tree, config)

    node = tree.root
    if not node.fully_expanded:
        return node

    while node.children:
        unexpanded = [c for c in node.children if not c.fully_expanded]
        if unexpanded:
            # Return the unexpanded child with the highest UCT score
            return max(unexpanded, key=lambda c: _uct_value(c, config))
        # All children expanded -- pick best UCT to traverse deeper
        node = max(node.children, key=lambda c: _uct_value(c, config))

    # Reached a fully-expanded leaf -- re-activate root for new drafts
    tree.root.fully_expanded = False
    return tree.root


def _uct_value(node: MCTSNode, config: MARSConfig) -> float:
    """Compute UCT value for a node."""
    if node.visit_count == 0:
        return float("inf")
    parent = node.parent
    if parent is None or parent.visit_count == 0:
        return node.q_value
    exploitation = node.q_value
    exploration = config.uct_constant * math.sqrt(math.log(parent.visit_count) / node.visit_count)
    return exploitation + exploration


def _update_expansion_status(tree: MCTSTree, config: MARSConfig) -> None:
    """Update fully_expanded flags for all nodes in the tree."""
    for node in tree.all_nodes:
        if node.is_root():
            # Root not fully expanded if: no children, or best stale
            if not node.children:
                node.fully_expanded = False
            elif tree._valid_nodes_since_best >= config.stale_threshold:
                node.fully_expanded = False
                tree._valid_nodes_since_best = 0  # reset counter
            else:
                node.fully_expanded = True
        elif node.is_buggy:
            node.fully_expanded = True
        elif node.is_valid():
            improvement_children = [c for c in node.children if c.action == "improve"]
            node.fully_expanded = len(improvement_children) >= config.max_improvements
        # else: non-root, non-valid, non-buggy nodes stay as-is


def backpropagate(node: MCTSNode, reward: float) -> None:
    """Backpropagate reward up the tree, updating Q values and visit counts.

    Uses incremental mean update:
        Q(s,a) <- Q(s,a) + (R - Q(s,a)) / N(s,a)
    """
    current: MCTSNode | None = node
    while current is not None:
        current.visit_count += 1
        current.q_value += (reward - current.q_value) / current.visit_count
        current = current.parent
