import os
import torch
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # --- Paths ---
    input_dir: str = "./input"
    metadata_dir: str = "./metadata"
    working_dir: str = "./working"
    # Specific directory for this idea's artifacts
    artifact_dir: str = "./working/idea_7"

    # --- Data Processing ---
    # Window size for the 1D CNN (must be odd to have a center frame)
    # 11 frames = t-5 to t+5 (approx +/- 0.5 seconds context)
    window_size: int = 11

    # Features to be used by the model
    # These correspond to the columns generated during feature engineering
    feature_cols: List[str] = field(
        default_factory=lambda: [
            "log_distance",  # Log1p(distance) for resolution near 0
            "relative_speed",  # |v1 - v2|
            "relative_acceleration",  # |a1 - a2|
            "closing_speed",  # Rate of change of distance
            "speed_1",  # Absolute speed player 1
            "speed_2",  # Absolute speed player 2 (0 if Ground)
            "acceleration_1",  # Absolute accel player 1
            "acceleration_2",  # Absolute accel player 2 (0 if Ground)
            "is_ground",  # Binary flag
            "orientation_cos_1",  # Cosine of orientation
            "orientation_sin_1",  # Sine of orientation
            "direction_cos_1",  # Cosine of movement direction
            "direction_sin_1",  # Sine of movement direction
        ]
    )

    # --- Model Architecture ---
    # Number of filters in the 1D Conv layers
    cnn_filters: int = 64
    # Kernel size for convolution
    cnn_kernel_size: int = 3
    # Hidden units in the dense head
    dense_hidden_units: int = 64
    dropout: float = 0.2

    # --- Training Hyperparameters ---
    seed: int = 42
    batch_size: int = 1024  # Large batch size for stability with Focal Loss
    learning_rate: float = 1e-3
    epochs: int = 10

    # Focal Loss parameters
    focal_alpha: float = 0.75  # Balance for class imbalance (favors minority class 1)
    focal_gamma: float = 2.0  # Focusing parameter for hard examples

    # Early Stopping
    patience: int = 3

    # Debugging/Development
    debug: bool = False  # If True, subsets data for quick testing
    debug_sample_size: int = 50000

    def __post_init__(self):
        # Ensure artifact directory exists
        os.makedirs(self.artifact_dir, exist_ok=True)


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
