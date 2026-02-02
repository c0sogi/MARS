import os
import torch
import random
import numpy as np


class Config:
    """
    Global configuration for the Parallel Low-Rank DCN-ResNet experiment.
    """

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Cache directory specifically for Idea 14 (safe for deterministic caching)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_14")

    # Submission output directory
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths (Pre-split)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Submission Files
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary writable directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    TARGET_COL = "Cover_Type"
    ID_COL = "Id"

    # The dataset typically contains classes 1-7.
    # The model output dimension will be 7.
    # Labels should be shifted by -1 (1-7 -> 0-6) during processing.
    NUM_CLASSES = 7

    # -------------------------------------------------------------------------
    # Model Architecture: Parallel Low-Rank DCN-ResNet
    # -------------------------------------------------------------------------
    # Branch 1: Low-Rank DCN
    DCN_RANK = 16  # Rank 'r' for the matrix decomposition W = U*V^T

    # Branch 2: Wide ResNet
    RESNET_WIDTH = 512  # Hidden dimension size
    RESNET_LAYERS = 2  # Depth of the ResNet backbone
    RESNET_DROPOUT = 0.1  # Dropout rate for regularization

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Budget-Aware Optimization
    BATCH_SIZE = 4096
    EPOCHS = 60

    # Optimizer Settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler: Cosine Annealing
    # Decays LR from LEARNING_RATE to ETA_MIN over T_MAX epochs
    T_MAX = EPOCHS
    ETA_MIN = 0.0

    # Early Stopping
    PATIENCE = 10  # Stop if validation accuracy doesn't improve

    # -------------------------------------------------------------------------
    # Hardware & System
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Suitable for 12 vCPUs

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # If True, the data loader should subsample the dataset for rapid prototyping
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 10000

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
