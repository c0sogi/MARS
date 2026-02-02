import os
import random
import numpy as np
import torch
from pathlib import Path


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")

    # Experiment specific working directory (Idea 8)
    WORKING_DIR = Path("./working/idea_8")

    # Metadata Paths
    TRAIN_METADATA_PATH = METADATA_DIR / "train.csv"
    VAL_METADATA_PATH = METADATA_DIR / "val.csv"
    TEST_METADATA_PATH = METADATA_DIR / "test.csv"

    # Final Submission Path (Home Directory)
    SUBMISSION_PATH = Path("./submission.csv")

    # Checkpoint Path
    BEST_MODEL_PATH = WORKING_DIR / "best_model.pth"

    # =========================================================================
    # Data Configuration
    # =========================================================================
    Z_DEPTH = 65  # Number of z-slices in the volume
    PATCH_SIZE = 256  # Spatial dimensions of training patches (256x256)

    # Normalization
    # We use global stats, but these are calculated/loaded during runtime.
    # Here we define the flag to use them.
    USE_GLOBAL_STATS = True

    # =========================================================================
    # Model Architecture (Full-Resolution Dilated U-Net)
    # =========================================================================
    IN_CHANNELS = 1  # Grayscale input
    PROJECTION_CHANNELS = 32  # Compressed depth feature space
    BACKBONE_CHANNELS = 64  # Base channel width for the dilated backbone
    DILATION_RATES = [1, 2, 4, 8, 16]  # Sequential dilation hierarchy
    GROUP_NORM_GROUPS = 8  # Number of groups for Group Normalization

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 8  # Adjusted for A100 memory with full resolution
    NUM_EPOCHS = 20  # Max epochs (Early stopping will likely intervene)
    LEARNING_RATE = 1e-4  # AdamW default
    WEIGHT_DECAY = 1e-2  # AdamW weight decay

    # Regularization
    TRAIN_SAMPLES_PER_EPOCH = 12000  # Explicit limit to prevent overfitting

    # Hardware
    NUM_WORKERS = 4  # Number of dataloader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Inference and Evaluation
    # =========================================================================
    # Test-Time Augmentation
    TTA_ENABLED = True

    # Threshold Optimization
    THRESHOLD_START = 0.1
    THRESHOLD_END = 0.9
    THRESHOLD_STEP = 0.01

    # Validation Grid
    VAL_TILE_SIZE = 256
    VAL_STRIDE = 256  # Non-overlapping for metric calculation speed/accuracy

    @staticmethod
    def setup():
        """Ensure necessary directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)


def seed_everything(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize environment
Config.setup()
