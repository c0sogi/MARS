import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    """
    Configuration class to store hyperparameters, file paths, and device settings.
    """

    def __init__(self, seed=42, debug=False, batch_size=128, epochs=25, lr=1e-3):
        """
        Initialize configuration with optional overrides.

        Args:
            seed (int): Random seed.
            debug (bool): If True, runs in debug mode with smaller subsets and fewer epochs.
            batch_size (int): Batch size for dataloaders.
            epochs (int): Maximum number of training epochs.
            lr (float): Learning rate.
        """
        # Reproducibility
        self.SEED = seed
        self.DEBUG = debug

        # Paths - Input
        self.INPUT_DIR = "./input"
        self.TRAIN_IMG_DIR = os.path.join(self.INPUT_DIR, "train")
        self.TEST_IMG_DIR = os.path.join(self.INPUT_DIR, "test")

        # Paths - Metadata
        self.METADATA_DIR = "./metadata"
        self.TRAIN_METADATA_PATH = os.path.join(self.METADATA_DIR, "train_metadata.csv")
        self.VAL_METADATA_PATH = os.path.join(self.METADATA_DIR, "val_metadata.csv")
        self.TEST_METADATA_PATH = os.path.join(self.METADATA_DIR, "test_metadata.csv")

        # Paths - Output
        self.WORKING_DIR = "./working"
        self.CACHE_DIR = os.path.join(self.WORKING_DIR, "idea_1")
        self.SUBMISSION_DIR = "./submission"
        self.SUBMISSION_PATH = os.path.join(self.SUBMISSION_DIR, "submission.csv")

        # Ensure output directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

        # Data Parameters
        self.IMAGE_SIZE = (32, 32)
        self.NUM_CLASSES = 1
        # If debug is True, limit dataset size for quick iteration
        self.DEBUG_SUBSET_SIZE = 500 if self.DEBUG else None

        # Training Hyperparameters
        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = 2 if self.DEBUG else epochs
        self.LEARNING_RATE = lr
        self.EARLY_STOPPING_PATIENCE = 5

        # Compute
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # Use 4 workers as we have 12 vCPUs available
        self.NUM_WORKERS = 4

    def __repr__(self):
        return (
            f"Config(seed={self.SEED}, debug={self.DEBUG}, device={self.DEVICE}, "
            f"epochs={self.NUM_EPOCHS}, batch_size={self.BATCH_SIZE})"
        )
