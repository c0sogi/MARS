import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration class for the Late-Fusion Time-Distributed CNN experiment.
    """

    # --- General Configuration ---
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use if DEBUG is True

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Directories
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # specific Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Specifications ---
    # Input data comes as (6, 273, 256)
    # Model expects (Batch, Time, Channels, Height, Width) -> (B, 6, 1, 273, 256)
    NUM_FRAMES = 6
    HEIGHT = 273
    WIDTH = 256
    CHANNELS = 1  # Single channel (intensity) per frame

    # --- Model Hyperparameters ---
    BACKBONE = "resnet50"
    PRETRAINED = True

    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 4  # Early stopping patience
    NUM_WORKERS = 4  # Number of dataloader workers

    # --- Compute ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initializes the experiment environment:
        1. Creates necessary working and submission directories.
        2. Sets fixed random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls._set_seed(cls.SEED)

    @staticmethod
    def _set_seed(seed):
        """
        Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic algorithms are used
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
