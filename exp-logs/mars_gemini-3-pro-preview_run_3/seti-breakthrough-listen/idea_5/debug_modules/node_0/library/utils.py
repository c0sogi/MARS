import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for the Siamese Spatial-Fusion Network.
    Contains file paths, model hyperparameters, and training settings.
    """

    # --- Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Output directory for checkpoints and cached data
    WORKING_DIR = "./working/idea_5"

    # --- Data Processing ---
    # Target image size for the EfficientNet backbone
    IMAGE_SIZE = (224, 224)
    # Number of workers for DataLoader
    NUM_WORKERS = 4

    # --- Model Architecture ---
    BACKBONE_NAME = "efficientnet_b0"
    PRETRAINED = True

    # --- Training Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    # Optimizer settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler settings (CosineAnnealingLR)
    T_MAX = 10

    # Regularization
    MIXUP_ALPHA = 0.2

    # --- Compute ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Creates the working directory if it does not exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
