import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Triple-Stream Wide-Body Network (TS-WBN) solution.
    Centralizes all hyperparameters, paths, and execution settings.
    """

    # ==========================================
    # 1. GENERAL SETTINGS
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to use a smaller subset of data for debugging
    DEBUG_SIZE = 100  # Number of samples to use when DEBUG is True

    # ==========================================
    # 2. FILE PATHS
    # ==========================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Directories (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (For caching intermediate data)
    WORKING_DIR = "./working/idea_47"
    CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. DATA SPECIFICATIONS
    # ==========================================
    IMAGE_SIZE = 75
    # Channels: Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
    NUM_CHANNELS = 3

    # ==========================================
    # 4. MODEL ARCHITECTURE
    # ==========================================
    # "Sustained Width Strategy": High channel capacity
    FILTERS = 128
    # "High Dropout": To prevent overfitting in the fusion head
    DROPOUT = 0.5

    # ==========================================
    # 5. TRAINING HYPERPARAMETERS
    # ==========================================
    NUM_FOLDS = 5
    BATCH_SIZE = 32
    EPOCHS = 50

    # Optimization Strategy: "Low and Slow"
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.0  # Standard Adam (not AdamW)

    # Scheduler: ReduceLROnPlateau
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 4
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 12

    # ==========================================
    # 6. HARDWARE & EXECUTION
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print("TS-WBN CONFIGURATION")
        print("=" * 40)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 40)


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_directories():
    """
    Ensures that the necessary working and submission directories exist.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
