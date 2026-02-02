import os
import random
import numpy as np
import torch
import pandas as pd


class Config:
    # ====================================================
    # Paths
    # ====================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working directories
    WORKING_DIR = "./working/idea_47"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ====================================================
    # Hyperparameters
    # ====================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a subset for debugging

    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    NUM_WORKERS = 4

    # Optimizer
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 5e-2  # Increased to 0.05 to combat early overfitting

    # Model / Architecture
    BACKBONE_NAME = "efficientnet_b2"
    IMAGE_SIZE = 260
    NUM_SLICES = 3  # Anchor + 2 boundary slices

    # Preprocessing
    # Lung Window: Level -600, Width 1500 => Range [-1350, 150]
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500
    WINDOW_MIN = WINDOW_LEVEL - (WINDOW_WIDTH // 2)
    WINDOW_MAX = WINDOW_LEVEL + (WINDOW_WIDTH // 2)

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ====================================================
    # Utilities
    # ====================================================
    @staticmethod
    def seed_everything(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_target_stats(load_cached_data=True):
        """
        Calculates or loads the Mean and Std of the FVC target from the training set.
        Used for Z-score normalization of the target.

        Args:
            load_cached_data (bool): If True, attempts to load from cache.

        Returns:
            tuple: (mean, std) of the FVC column in train.csv
        """
        cache_path = os.path.join(Config.CACHE_DIR, "target_stats.npy")

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(cache_path):
            try:
                stats = np.load(cache_path)
                return stats[0], stats[1]
            except Exception:
                pass  # Fallback to computation if load fails

        # 2. Compute from scratch
        if not os.path.exists(Config.TRAIN_META_PATH):
            raise FileNotFoundError(
                f"Training metadata not found at {Config.TRAIN_META_PATH}"
            )

        df = pd.read_csv(Config.TRAIN_META_PATH)
        target_mean = df["FVC"].mean()
        target_std = df["FVC"].std()

        # 3. Save to cache
        np.save(cache_path, np.array([target_mean, target_std]))

        return target_mean, target_std


# Initialize seed immediately on import for consistency
Config.seed_everything(Config.SEED)
