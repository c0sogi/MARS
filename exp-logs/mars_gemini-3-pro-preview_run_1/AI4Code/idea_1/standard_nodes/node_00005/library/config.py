import os
import torch
from dataclasses import dataclass


@dataclass
class Config:
    """
    Global configuration for the Semantic Anchor Classifier project.
    """

    # ==============================
    # General Settings
    # ==============================
    seed: int = 42
    debug: bool = False
    debug_sample_size: int = 1000  # Number of notebooks to use when debug=True
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ==============================
    # Paths
    # ==============================
    input_dir: str = "./input"
    metadata_dir: str = "./metadata"
    working_dir: str = "./working/idea_1"
    submission_dir: str = "./submission"

    # Metadata File Paths
    train_metadata_path: str = os.path.join(metadata_dir, "train_metadata.csv")
    val_metadata_path: str = os.path.join(metadata_dir, "val_metadata.csv")
    test_metadata_path: str = os.path.join(metadata_dir, "test_metadata.csv")

    # Cache File Paths
    # Using Parquet for structured data caching as requested
    train_cache_path: str = os.path.join(working_dir, "train_features.parquet")
    val_cache_path: str = os.path.join(working_dir, "val_features.parquet")
    test_cache_path: str = os.path.join(working_dir, "test_features.parquet")

    # Model Checkpoint Path
    model_save_path: str = os.path.join(working_dir, "best_model.pth")

    # Submission Path
    submission_path: str = os.path.join(submission_dir, "submission.csv")

    # ==============================
    # Model Hyperparameters
    # ==============================
    # Frozen backbone model from Sentence Transformers
    backbone_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Max token length for cell content
    max_length: int = 128

    # Dimension of the embeddings output by the backbone (384 for MiniLM-L6-v2)
    input_dim: int = 384

    # Hidden dimension for the projection head (MLP)
    hidden_dim: int = 512

    # Dropout rate for the projection head
    dropout: float = 0.2

    # ==============================
    # Training Settings
    # ==============================
    # Batch size refers to number of notebooks per batch
    train_batch_size: int = 8
    val_batch_size: int = 16

    # Optimizer settings
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # Training loop controls
    epochs: int = 10
    early_stopping_patience: int = 3

    def __post_init__(self):
        """
        Setup directories and perform basic validation after initialization.
        """
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

        if self.debug:
            print(
                f"Configuration initialized in DEBUG mode (sample_size={self.debug_sample_size})"
            )
