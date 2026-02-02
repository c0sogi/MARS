import os
import random
import numpy as np
import torch
from pathlib import Path


class Config:
    """
    Configuration class for the ESDN-PCH (Extended Sequential Dilated Network
    with Parallel Context Head) solution.
    """

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = Path("./input")
    METADATA_DIR = Path("./metadata")
    WORKING_DIR = Path("./working")

    # Specific cache directory for this idea iteration
    CACHE_DIR = WORKING_DIR / "idea_9"

    # Output paths
    # Submission must be in the home directory
    SUBMISSION_PATH = Path("submission.csv")
    MODEL_PATH = CACHE_DIR / "best_model.pth"
    NORMALIZATION_STATS_PATH = CACHE_DIR / "normalization_stats.npy"
    THRESHOLD_PATH = CACHE_DIR / "threshold.txt"

    # Ensure cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    Z_DIM = 65
    PATCH_SIZE = 256

    # Normalization: Use global statistics (mean/std) derived from training set
    # to preserve physical density signals.
    USE_GLOBAL_STATS = True

    # -------------------------------------------------------------------------
    # Model Architecture: ESDN-PCH
    # -------------------------------------------------------------------------
    # Lean channel width to allow for a healthy Batch Size of 32
    BACKBONE_CHANNELS = 32

    # Extended dilation hierarchy to capture long-range stroke continuity
    # r = 1, 2, 4, 8, 16, 32
    BACKBONE_DILATION_RATES = [1, 2, 4, 8, 16, 32]

    # Parallel Context Head (PCH) dilation rates
    # Aggregates local and broad context at the network head
    HEAD_DILATION_RATES = [1, 6, 12, 18]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42

    # Priority: High Batch Size for Stable BN
    BATCH_SIZE = 32

    LEARNING_RATE = 1e-3
    EPOCHS = 25

    # Regularization: Explicitly limit samples per epoch to prevent overfitting
    # to the specific noise patterns of a single fragment.
    MAX_PATCHES_PER_EPOCH = 12000

    # Optimization
    EARLY_STOPPING_PATIENCE = 5
    WEIGHT_DECAY = 1e-4

    # -------------------------------------------------------------------------
    # Inference Configuration
    # -------------------------------------------------------------------------
    # Test-Time Augmentation (TTA)
    TTA_FLIPS = True  # Horizontal and Vertical flips
    TTA_ROTATIONS = True  # 90-degree rotations

    # Dynamic Threshold Search
    THRESHOLD_SEARCH_START = 0.2
    THRESHOLD_SEARCH_END = 0.8
    THRESHOLD_SEARCH_STEP = 0.01

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the seed for reproducibility across random, numpy, and torch.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
