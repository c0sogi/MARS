import os
import torch
import numpy as np
import random


class Config:
    """
    Central configuration for the LSE-I-CNN solution.
    Handles hyperparameters, file paths, and reproducibility settings.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_62"

    # Derived Output Directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Raw Data Filenames
    TRAIN_JSON = "train.json"
    TEST_JSON = "test.json"

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 75
    INPUT_CHANNELS = 3  # Channel 0: HH, Channel 1: HV, Channel 2: Average

    # -------------------------------------------------------------------------
    # Model Architecture (LSE-I-CNN)
    # -------------------------------------------------------------------------
    # Backbone: Plain CNN with 4 stages (Early Expansion)
    # Width Strategy: 64 -> 128 -> 128 -> 128
    BACKBONE_CHANNELS = [64, 128, 128, 128]

    # Readout: Extract features from Stage 3 and Stage 4
    # Indices are 0-based, so 2 corresponds to Stage 3, 3 to Stage 4
    READOUT_STAGES = [2, 3]

    # Projection dimension for the LSE/SoftMin operations
    # The final feature vector size will be:
    #   len(READOUT_STAGES) * PROJECTION_DIM * 2 (Positive + Negative Polarity)
    #   2 * 64 * 2 = 256 features
    PROJECTION_DIM = 64

    # Classification Head
    # Input to linear layer is 256 (features) + 1 (inc_angle) = 257
    FC_DIM = 256
    DROPOUT = 0.5

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    N_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 75
    PATIENCE = 12  # For Early Stopping

    # Optimizer: AdamW
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # -------------------------------------------------------------------------
    # Compute & Hardware
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary working directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print(f"CONFIG: {cls.__name__}")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<20}: {v}")
        print("=" * 40)
