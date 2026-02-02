import os
import torch
import numpy as np
import random


def set_seed(seed=42):
    """
    Sets the random seed for reproducible results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SIZE = 100  # Number of samples to use when DEBUG is True

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_50"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache File (for processed data)
    PROCESSED_DATA_PATH = os.path.join(WORK_DIR, "processed_data.npz")

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 75
    IN_CHANNELS = 3  # Band 1, Band 2, Mean((B1+B2)/2)

    # Normalization: Global Min-Max Scaling (computed on full training set)
    # Values can exceed [0, 1] in test set (No Hard Clipping)
    NORM_METHOD = "global_min_max"

    # --------------------------------------------------------------------------
    # Model Architecture (RDP-WBN)
    # --------------------------------------------------------------------------
    BASE_FILTERS = 128  # Wide-Body Backbone
    DROPOUT_RATE = 0.5  # High dropout for regularization
    NUM_CLASSES = 1  # Binary classification

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    NUM_FOLDS = 5
    BATCH_SIZE = 64  # Optimized for A100 GPU
    LEARNING_RATE = 2e-4  # "Low and Slow" strategy
    NUM_EPOCHS = 100  # Sufficient duration for convergence
    PATIENCE = 15  # Early stopping patience

    OPTIMIZER = "Adam"  # Standard Adam (not AdamW)
    WEIGHT_DECAY = 0  # Rely on Dropout for regularization

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-6

    # --------------------------------------------------------------------------
    # Augmentation Strategy
    # --------------------------------------------------------------------------
    # Rotational Invariance: 0, 90, 180, 270 degrees
    AUG_ROTATION = True
    # Horizontal Flip
    AUG_HFLIP = True
    # Prohibited Augmentations
    AUG_VFLIP = False
    AUG_MIXUP = False

    # --------------------------------------------------------------------------
    # Compute Configuration
    # --------------------------------------------------------------------------
    NUM_WORKERS = 4  # 12 vCPUs available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initializes the environment: creates directories and sets seeds.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        set_seed(cls.SEED)


# Execute setup immediately upon import to ensure environment is ready
Config.setup()
