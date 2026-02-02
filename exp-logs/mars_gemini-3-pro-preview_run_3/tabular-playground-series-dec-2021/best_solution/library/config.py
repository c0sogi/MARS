import os
import torch
from dataclasses import dataclass


@dataclass
class PathConfig:
    """
    Defines all file paths for the project.
    Uses metadata parquet files for data loading and working directory for caching.
    """

    # Base Directories
    input_dir: str = "./input"
    metadata_dir: str = "./metadata"
    working_dir: str = "./working/idea_31"
    submission_dir: str = "./submission"

    # Input Metadata Files (Parquet) - Source of Truth
    train_parquet: str = os.path.join(metadata_dir, "train.parquet")
    val_parquet: str = os.path.join(metadata_dir, "val.parquet")
    test_parquet: str = os.path.join(metadata_dir, "test.parquet")

    # Sample Submission
    sample_submission: str = os.path.join(input_dir, "sample_submission.csv")

    # Cache Files (Numpy) - For deterministic data processing
    train_X_path: str = os.path.join(working_dir, "train_X.npy")
    train_y_path: str = os.path.join(working_dir, "train_y.npy")
    val_X_path: str = os.path.join(working_dir, "val_X.npy")
    val_y_path: str = os.path.join(working_dir, "val_y.npy")
    test_X_path: str = os.path.join(working_dir, "test_X.npy")
    test_ids_path: str = os.path.join(working_dir, "test_ids.npy")

    # Model Artifacts
    model_save_path: str = os.path.join(working_dir, "best_model.pth")

    # Final Submission
    submission_path: str = os.path.join(submission_dir, "submission.csv")

    def __post_init__(self):
        """Ensure necessary writable directories exist."""
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)


@dataclass
class ModelConfig:
    """
    Hyperparameters for the Deep Parallel Vector-DCN-ResNet.
    """

    # Input Dimensions
    # Original: 54
    # Engineered: + Aspect_Sin, Aspect_Cos, Euclidean_Hydro, Abs_Hydro_Elev, Mean_Amenities (+5)
    input_dim: int = 59

    # Architecture
    hidden_dim: int = 512
    num_resnet_blocks: int = 4  # Deep Full Pre-Activation Backbone
    num_cross_layers: int = 3  # Vector-based Cross Network depth
    dropout_rate: float = 0.2

    # Classification
    num_classes: int = 7  # Cover_Type classes 1-7 (mapped to 0-6 internally)

    # Initialization
    cross_layer_init_std: float = 1e-4  # Near-zero init for DCN branch


@dataclass
class TrainConfig:
    """
    Training hyperparameters and optimization settings.
    """

    # Optimization
    batch_size: int = 4096
    epochs: int = 60
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2  # For AdamW

    # Scheduler (ReduceLROnPlateau)
    scheduler_factor: float = 0.1
    scheduler_patience: int = 5

    # Early Stopping
    early_stopping_patience: int = 10

    # System & Reproducibility
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    debug: bool = False
    debug_sample_size: int = 10000


class Config:
    """
    Global configuration container.
    """

    paths = PathConfig()
    model = ModelConfig()
    train = TrainConfig()


# Expose a single instance for import
config = Config()
