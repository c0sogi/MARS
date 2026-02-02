import os
import torch
import numpy as np
import random


class Config:
    # --- System & Reproducibility ---
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Conservative number for data loading

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this iteration (Idea 6)
    WORKING_DIR = "./working/idea_6"

    # Subdirectories for organization
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")

    # Metadata Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Submission Path (Home directory as per task description)
    SUBMISSION_PATH = "./submission.csv"

    # --- Data Dimensions ---
    Z_DIM = 65  # Number of slices in the z-direction
    PATCH_HEIGHT = 512
    PATCH_WIDTH = 512

    # --- Model Hyperparameters ---
    BATCH_SIZE = 2
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    POS_WEIGHT = 2.0  # Weight for positive class in BCE loss

    # Model Architecture Params
    IN_CHANNELS = 65  # Input channels (depth slices)
    STEM_CHANNELS = 64  # Channels after projection

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
