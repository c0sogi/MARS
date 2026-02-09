"""Efficiency-guided reward R(v) per Equations 3-4 of the MARS paper."""

from __future__ import annotations

from mars.config import MARSConfig
from mars.mcts.tree import MCTSNode


def compute_reward(
    node: MCTSNode,
    all_nodes: list[MCTSNode],
    config: MARSConfig,
    lower_is_better: bool,
) -> float:
    """Compute efficiency-guided reward R(v) per Equations 3-4.

    Eq 3 -- Global normalised score:
        G(v) = 0.5                              if M_max == M_min
        G(v) = (M(v) - M_min) / (M_max - M_min) otherwise

    Eq 4 -- Efficiency-guided reward:
        R(v) = G(v) * (t(v) / L(v))^w

    where w = config.reward_weight, L(v) = config.script_timeout.
    When *lower_is_better*, metrics are negated before normalisation so
    that a higher G still corresponds to a better solution.
    """
    valid_nodes = [n for n in all_nodes if n.is_valid()]
    if not valid_nodes or not node.is_valid():
        return 0.0

    metrics: list[float] = []
    for n in valid_nodes:
        m = n.metric_value
        assert m is not None
        metrics.append(-m if lower_is_better else m)

    node_metric: float = (
        -node.metric_value if lower_is_better else node.metric_value  # type: ignore[operator]
    )

    m_max = max(metrics)
    m_min = min(metrics)

    # Eq 3: Global normalised score
    if m_max == m_min:
        g = 0.5
    else:
        g = (node_metric - m_min) / (m_max - m_min)

    # Eq 4: Efficiency-guided reward
    exec_time = node.execution_time if node.execution_time is not None else float(config.script_timeout)
    time_ratio = exec_time / config.script_timeout
    # Clamp to avoid issues with very small times
    time_ratio = max(time_ratio, 1e-6)

    r: float = g * (time_ratio**config.reward_weight)
    return r
