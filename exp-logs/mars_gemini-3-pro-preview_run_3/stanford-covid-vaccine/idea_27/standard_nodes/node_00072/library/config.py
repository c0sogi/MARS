import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Deep Input-Aware Channel-Gated BiGRU pipeline.
    Centralizes all hyperparameters, paths, and environment settings.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Parquet format for efficient loading)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.parquet")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Cache Files (Numpy format for processed tensors)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # =========================================================================
    # Data Dimensions
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68
    NUM_TARGETS = 5

    # Input Channels
    # 4 (Nucleotide: A, G, C, U)
    # + 3 (Structure: (, ), .)
    # + 7 (Loop Type: S, M, I, B, H, E, X)
    INPUT_CHANNELS = 14

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Strategy: Deep Input-Aware Channel-Gated BiGRU
    HIDDEN_DIM = 384
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 3

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 16
    LR = 1e-3
    MAX_GRAD_NORM = 1.0
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # Scheduler Settings (Cosine Annealing)
    MIN_LR = 1e-6

    # =========================================================================
    # System / Reproducibility
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging Flags
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary working directories.
        2. Sets random seeds for reproducibility across random, numpy, and torch.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
