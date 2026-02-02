import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run with a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Input data (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Generated previously)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Output directories
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.bin")

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 96

    # Multi-Sample Dropout Settings
    USE_MULTI_SAMPLE_DROPOUT = True
    DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    EPOCHS = 3
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32

    # Optimizer
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    SCHEDULER_TYPE = "linear"  # or 'cosine'
    WARMUP_RATIO = 0.1

    # Loss Function
    USE_SOFT_JACCARD = True
    SOFT_JACCARD_WEIGHT = 0.5  # Weight for the soft jaccard component in the loss

    # Data Strategy
    FILTER_NEUTRAL_TRAIN = True  # Exclude 'neutral' samples from training

    # =========================================================================
    # Hardware & System
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary working and submission directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.seed_everything(cls.SEED)

    @staticmethod
    def seed_everything(seed=42):
        """
        Sets the seed for all random number generators to ensure reproducibility.
        """
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
