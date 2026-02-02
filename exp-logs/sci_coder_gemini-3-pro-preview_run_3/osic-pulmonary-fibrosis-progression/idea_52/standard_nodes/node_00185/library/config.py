import os
import torch
import random
import numpy as np


class Config:
    """
    Global configuration for the Affine-Isolated Latent-Residual Network (AILR-Net).
    """

    # --- File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Raw Data Directories
    TRAIN_DICOM_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DICOM_DIR = os.path.join(INPUT_DIR, "test")

    # Working Directories
    # Specific folder for this solution iteration
    IDEA_NAME = "idea_52"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # --- Data Preprocessing ---
    # Image Parameters
    IMG_SIZE = 260
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500
    NUM_SLICES = 3  # Anchor + 2 boundary slices

    # Normalization Statistics (Global Statistics from EDA)
    # Mean and Std of FVC across the entire dataset
    TARGET_MEAN = 2654.65
    TARGET_STD = 801.70

    # --- Model Architecture ---
    MODEL_NAME = "AILR-Net"
    BACKBONE = "efficientnet_b2"
    LATENT_DIM = 128
    # Explicitly 0.0 for Stream B to preserve residual signal
    DROPOUT = 0.0

    # --- Training Hyperparameters ---
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 50

    # Optimization
    # Differential Learning Rates
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 0.01

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Metric & Post-Processing ---
    SIGMA_MIN = 70.0
    MAX_ERROR = 1000.0

    @classmethod
    def setup(cls, seed=None):
        """
        Initializes the working environment.
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set Seed
        effective_seed = seed if seed is not None else cls.SEED
        random.seed(effective_seed)
        np.random.seed(effective_seed)
        torch.manual_seed(effective_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(effective_seed)
            torch.cuda.manual_seed_all(effective_seed)
            # Ensure deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Environment setup complete. Working directory: {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}, Seed: {effective_seed}")
