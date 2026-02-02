import os
import random
import numpy as np
import torch
from pathlib import Path


class Config:
    # --- Paths ---
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")

    # Working directory for this specific idea/iteration
    # Using idea_7 as requested for caching and outputs
    WORKING_DIR = Path("./working/idea_7")

    # Subdirectories for organized outputs
    CHECKPOINT_DIR = WORKING_DIR / "checkpoints"
    PREDICTION_DIR = WORKING_DIR / "predictions"
    CACHE_DIR = WORKING_DIR  # For cached numpy/parquet files

    # Metadata files
    TRAIN_METADATA = METADATA_DIR / "train.csv"
    VAL_METADATA = METADATA_DIR / "val.csv"
    TEST_METADATA = METADATA_DIR / "test.csv"

    # Submission file
    SUBMISSION_PATH = Path("./submission.csv")

    # --- Data Parameters ---
    Z_DIM = 65  # Depth of the 3D volume
    PATCH_SIZE = 256  # Height and Width of training patches
    STRIDE = PATCH_SIZE // 2  # Overlap for validation/inference tiling

    # --- Model Architecture Parameters ---
    # Input channels = Z_DIM because we treat depth as channels for the initial 1x1 projection
    IN_CHANNELS = 65
    PROJECTION_DIM = 32  # Output channels after the learnable 2.5D projection
    BACKBONE_WIDTH = 32  # Channel width for the dilated blocks
    DILATION_RATES = [1, 2, 4, 8]  # Hierarchical dilation rates

    # --- Training Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20  # Maximum epochs, controlled by early stopping
    PATIENCE = 4  # Strict patience for early stopping
    NUM_WORKERS = 4  # For DataLoader

    # --- Inference Parameters ---
    THRESHOLD_SEARCH_STEPS = 50  # Number of steps for dynamic threshold tuning
    TTA_FLIPS = True  # Test-Time Augmentation: Flips
    TTA_ROTATIONS = True  # Test-Time Augmentation: Rotations (90, 180, 270)

    # --- Device ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTION_DIR, exist_ok=True)

        # Set seeds
        cls.seed_everything(cls.SEED)

    @staticmethod
    def seed_everything(seed):
        """
        Sets seeds for reproducibility.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Initialize setup immediately when module is imported to ensure dirs exist
Config.setup()
