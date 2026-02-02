import os
import torch
from dataclasses import dataclass, field
from typing import List

# Global Random Seed for Reproducibility
SEED = 42


@dataclass
class ModelConfig:
    """
    Configuration for the Hybrid Geometric-Attention Network.

    Attributes:
        hidden_dim: Dimension of atom and edge embeddings.
        num_mp_layers: Number of Directional Message Passing layers (Backbone).
        num_transformer_layers: Number of Graph Transformer layers (Global Interaction).
        num_heads: Number of attention heads in the Transformer.
        cutoff: Radial cutoff distance in Angstroms for graph construction.
        num_rbf: Number of Radial Basis Functions for edge distance encoding.
        num_sbf: Number of Spherical Basis Functions for triplet angle encoding.
        dropout: Dropout probability (set to 0.0 as per strategy).
        num_atom_types: Maximum number of atom types (e.g., H, C, N, O, F).
        coupling_types: List of scalar coupling types to predict.
    """

    hidden_dim: int = 256
    num_mp_layers: int = 4
    num_transformer_layers: int = 2
    num_heads: int = 8
    cutoff: float = 5.0
    num_rbf: int = 64
    num_sbf: int = 32
    dropout: float = 0.0
    num_atom_types: int = 10
    coupling_types: List[str] = field(
        default_factory=lambda: [
            "1JHC",
            "2JHC",
            "3JHC",
            "1JHN",
            "2JHN",
            "3JHN",
            "2JHH",
            "3JHH",
        ]
    )


@dataclass
class TrainConfig:
    """
    Configuration for training, evaluation, and system paths.

    Attributes:
        input_dir: Root directory for input data.
        metadata_dir: Directory containing processed metadata CSVs.
        working_dir: Directory for caching intermediate files and artifacts.
        submission_path: Path to save the final submission file.
        model_path: Path to save the best model checkpoint.
        batch_size: Number of samples per batch.
        epochs: Maximum number of training epochs.
        learning_rate: Peak learning rate for the scheduler.
        weight_decay: L2 regularization factor.
        warmup_epochs: Number of epochs for linear warmup.
        patience: Patience for Early Stopping.
        num_workers: Number of DataLoader workers.
        device: Computation device ('cuda' or 'cpu').
        debug: If True, uses a small subset of data for rapid testing.
        debug_samples: Number of samples to use when debug is True.
    """

    # Directory Paths
    input_dir: str = "./input"
    metadata_dir: str = "./metadata"
    working_dir: str = "./working/idea_4"
    submission_path: str = "./submission/submission.csv"
    model_path: str = "./working/idea_4/best_model.pt"

    # Optimization Hyperparameters
    batch_size: int = 96
    epochs: int = 40
    learning_rate: float = 5e-4
    weight_decay: float = 1e-6
    warmup_epochs: int = 3
    patience: int = 8

    # System Settings
    num_workers: int = 8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging
    debug: bool = False
    debug_samples: int = 10000

    def __post_init__(self):
        """Ensure necessary directories exist."""
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.submission_path), exist_ok=True)
