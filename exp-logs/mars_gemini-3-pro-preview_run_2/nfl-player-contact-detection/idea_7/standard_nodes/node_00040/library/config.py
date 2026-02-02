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
    artifact_dir: str = "./working/idea_7_v2"

    # --- Data Processing ---
    # Window size for the MLP (t-5 to t+5)
    window_size: int = 11

    # Path for the fitted scaler
    scaler_path: str = "./working/idea_7/scaler.joblib"

    # --- Model Architecture ---
    # MLP Architecture
    hidden_units: List[int] = field(default_factory=lambda: [512, 256, 128])
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
