import os
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """
    Configuration for the Heterogeneous-Embedding Wide-Stream Residual BiGRU model.
    """

    # ==============================
    # File Paths & Directories
    # ==============================
    input_dir: str = "./input"
    metadata_dir: str = "./metadata"
    working_dir: str = "./working/idea_39"

    # Metadata files (Pre-split parquet files)
    train_file: str = os.path.join(metadata_dir, "train.parquet")
    val_file: str = os.path.join(metadata_dir, "val.parquet")
    test_file: str = os.path.join(metadata_dir, "test.parquet")

    # Submission template
    sample_submission_file: str = os.path.join(input_dir, "sample_submission.csv")

    # Output path for the final submission
    submission_file: str = os.path.join(working_dir, "submission.csv")

    # ==============================
    # Data Dimensions & Features
    # ==============================
    seq_len: int = 107
    pred_len: int = 68

    # Vocabularies
    vocab_size: int = 4  # A, G, C, U
    loop_types_size: int = 7  # S, M, I, B, H, E, X

    # ==============================
    # Model Architecture
    # ==============================
    # Heterogeneous Embedding Dimensions (Proportional Feature Embedding)
    emb_dim_seq: int = 128  # Atomic Sequence Identity
    emb_dim_loop: int = 64  # Predicted Loop Type
    emb_dim_dist: int = 64  # Signed Sinusoidal Pairing Distance

    # Wide-Stream Residual Backbone
    hidden_dim: int = 512  # Stream width W
    num_layers: int = 6  # Number of Residual Blocks
    dropout: float = 0.2  # Inter-layer dropout

    # ==============================
    # Training & Optimization
    # ==============================
    # Targets: Only train on the scored columns to reduce noise
    target_cols: List[str] = field(
        default_factory=lambda: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    )

    batch_size: int = 64
    num_epochs: int = 20

    # Optimizer settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4  # Low weight decay to preserve recurrent signal
    max_grad_norm: float = 1.0  # Gradient clipping for stability

    # ==============================
    # System & Reproducibility
    # ==============================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    seed: int = 42

    def __post_init__(self):
        """
        Ensure the working directory exists upon initialization.
        """
        os.makedirs(self.working_dir, exist_ok=True)
