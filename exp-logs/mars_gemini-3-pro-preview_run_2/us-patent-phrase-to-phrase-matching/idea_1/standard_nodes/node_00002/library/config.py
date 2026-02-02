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
    # Pre-trained Sentence Transformer backbone
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    # Text processing
    MAX_LENGTH = 128  # Max token length for the tokenizer

    # --- Compute Configuration ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Batch size for generating embeddings (inference mode)
    BATCH_SIZE = 128
    NUM_WORKERS = 4

    # --- Training/Algorithm Hyperparameters ---
    # Regularization strength for the Ridge Regression head
    # Higher values specify stronger regularization.
    RIDGE_ALPHA = 1.0

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
