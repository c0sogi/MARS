"""Global configuration and hyperparameters for MARS framework."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MARSConfig:
    """Configuration for MARS framework.

    Hyperparameters from Section 5.1 of the paper.
    """

    # MCTS parameters
    max_lessons: int = 30  # K_m: max lessons in each pool
    max_debug_attempts: int = 10  # N_d: max debugging attempts per node
    max_improvements: int = 2  # N_i: branching factor for valid nodes
    reward_weight: float = -0.07  # w: efficiency penalty weight
    uct_constant: float = 1.414  # c_uct: UCT exploration constant (sqrt(2))
    stale_threshold: int = 3  # n_s: re-draft after stale best

    # Model search
    num_model_candidates: int = 5  # K_a: number of architecture candidates

    # Execution
    exec_timeout: int = 86400  # 24h wall-clock budget (seconds)
    script_timeout: int = 14400  # 4h per script execution (seconds)

    # LLM
    model_name: str = "gemini-2.5-pro"
    temperature: float = 1.0

    # Paths
    work_dir: str = "./workspace"
    input_dir: str = "./input"

    # Internal tracking
    node_counter: int = field(default=0, repr=False)

    def next_node_id(self) -> str:
        """Generate next sequential node ID."""
        node_id = f"node_{self.node_counter:05d}"
        self.node_counter += 1
        return node_id
