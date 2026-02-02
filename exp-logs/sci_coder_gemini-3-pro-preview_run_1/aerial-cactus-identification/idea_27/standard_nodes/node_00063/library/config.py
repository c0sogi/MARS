import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration for the Heterogeneous Quality-Calibrated Stacking (HQCS) pipeline.
    """

    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    PROJECT_NAME = "HQCS_Cactus"
    DEBUG = False  # Set to True to use a small subset of data for debugging

    # --------------------------------------------------------------------------
    # Hardware & Compute
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for available vCPUs

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Paths (Generated Pre-split)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directories (Writeable)
    # Using 'idea_27' as specified for this run
    WORKING_DIR = "./working/idea_27"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Parameters
    # --------------------------------------------------------------------------
    IMG_SIZE = 32
    NUM_CLASSES = 1
    INPUT_CHANNELS = 3

    # Normalization Statistics (Calculated from Data Analysis)
    # Mean: R=128.37, G=115.25, B=119.40 -> Normalized by 255
    MEAN = [0.5034, 0.4520, 0.4682]
    # Std: R=38.60, G=35.68, B=39.15 -> Normalized by 255
    STD = [0.1514, 0.1399, 0.1535]

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    N_FOLDS = 5
    EPOCHS = 30
    SWA_START_EPOCH = 20
    BATCH_SIZE = 128

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # For AdamW

    # Regularization & Augmentation
    MIXUP_ALPHA = 0.2

    # Loss Configuration
    AUX_WEIGHT = 0.5  # Weight for the file-size regression auxiliary task

    @classmethod
    def initialize(cls):
        """
        Sets up the environment: creates directories and sets random seeds.
        """
        cls._setup_directories()
        cls._set_seed(cls.SEED)

    @classmethod
    def _setup_directories(cls):
        """Creates necessary working directories if they don't exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def _set_seed(seed):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
