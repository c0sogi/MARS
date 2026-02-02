import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets fixed random seeds for random, numpy, torch, and CUDA
    to guarantee fully reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    """
    Central configuration class for the Phrase Similarity Task.
    Implements settings for the Frozen Bi-Encoder + Linear Regression approach.
    """

    # --- General Configuration ---
    SEED = 42
    # Flag to control dataset size for debugging/fast prototyping
    DEBUG = False
    DEBUG_SIZE = 500  # Number of samples to use if DEBUG is True

    # --- File Paths ---
    # Input metadata paths (read-only)
    METADATA_DIR = "./metadata"
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output and Cache paths
    WORKING_DIR = "./working"
    # Specific cache directory for Idea 1 (Frozen Bi-Encoder)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Model Configuration ---
    # Pre-trained Transformer backbone (Cross-Encoder)
    MODEL_NAME = "microsoft/deberta-v3-small"

    # Text processing
    MAX_LENGTH = 128  # Max token length for the tokenizer

    # --- Compute Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2

    # --- Training/Algorithm Hyperparameters ---
    TRAIN_BATCH_SIZE = 32
    VAL_BATCH_SIZE = 64
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1

    # Caching behavior
    LOAD_CACHED_DATA = True

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary output directories.
        2. Set random seeds.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        set_seed(cls.SEED)
