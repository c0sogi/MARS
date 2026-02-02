import os
import sys
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the Identity-Aware Symmetric Dual-Axis Network (IAS-DAN) pipeline.
    Handles hyperparameters, file paths, and environment setup.
    """

    # ==========================================
    # Experiment Identifiers
    # ==========================================
    EXPERIMENT_NAME = "idea_25"
    SEED = 42
    DEBUG = False

    # ==========================================
    # Directory Paths
    # ==========================================
    # Read-Only Input Directories
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSV Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Writable Working Directories
    # Using 'idea_25' as requested to isolate this solution's artifacts
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Weights
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    IMAGE_SIZE = 224
    NUM_SLABS = 3  # Tri-Slab configuration
    SLAB_OVERLAP = 0.15  # 15% overlap between slabs

    # Tabular Features
    TABULAR_FEATURES = ["Age", "Sex", "SmokingStatus", "Percent"]

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    BACKBONE = "efficientnet_b0"
    BACKBONE_PRETRAINED = True
    FEATURE_DIM = 1280  # Native dimensionality of EfficientNet-B0 (no bottleneck)
    HIDDEN_DIM = 512  # Internal dimension for tabular MLP

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 8  # Strict early stopping
    NUM_WORKERS = 4  # Utilizing available vCPUs

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Metric & Loss Constants
    # ==========================================
    Q_SIGMA_THRESHOLD = 70.0  # Confidence clipping (ml)
    ERROR_THRESHOLD = 1000.0  # Error clipping (ml)

    @classmethod
    def initialize(cls):
        """
        Initializes the experiment environment:
        1. Creates all necessary writable directories.
        2. Sets fixed random seeds for reproducibility.
        """
        # Create directories
        directories = [
            cls.WORKING_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.LOG_DIR,
            cls.SUBMISSION_DIR,
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        # Set reproducibility seeds
        cls._set_seed()

    @classmethod
    def _set_seed(cls):
        """Sets fixed seeds for random, numpy, and torch."""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def set_debug_mode(cls, debug=True):
        """
        Adjusts hyperparameters for debugging/testing purposes.

        Args:
            debug (bool): If True, reduces epochs and dataset usage.
        """
        cls.DEBUG = debug
        if debug:
            cls.EPOCHS = 2
            cls.BATCH_SIZE = 8
            cls.NUM_WORKERS = 0  # Avoid multiprocessing overhead in debug
            # Additional debug flags can be handled in data/training loops
