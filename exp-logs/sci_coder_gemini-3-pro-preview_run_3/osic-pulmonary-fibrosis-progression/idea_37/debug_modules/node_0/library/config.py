import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration module for the Regularized Cascaded Output-Space Residual Network (RCOSR-Net).
    Centralizes all hyperparameters, file paths, and constants.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run with a small subset of data
    DEBUG_SAMPLE_SIZE = 20  # Number of patients to use when DEBUG is True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment/idea
    WORKING_DIR = "./working/idea_37"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # -------------------------------------------------------------------------
    # Data Preprocessing Parameters
    # -------------------------------------------------------------------------
    # CT Scan Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Image Selection and Resizing
    IMG_SIZE = 260  # Native resolution for EfficientNet-B2
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices

    # Normalization Statistics (Derived from EDA)
    # Target Variable: FVC
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Input Feature: Age
    AGE_MEAN = 67.5825
    AGE_STD = 6.6259

    # Time Scaling
    # Relative Time = (Weeks - Baseline_Week) * TIME_SCALE
    TIME_SCALE = 0.01

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b2"
    PRETRAINED = True

    # Clinical Stream (Stream A)
    # Inputs: Baseline_FVC, Relative_Time, Age, Sex, SmokingStatus
    # Dimensions: 1 + 1 + 1 + 1 + 1 = 5
    CLINICAL_INPUT_DIM = 5
    CLINICAL_HIDDEN_DIM = 128
    CLINICAL_LATENT_DIM = 64

    # Visual Stream (Stream B)
    DROPOUT_RATE = 0.2

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 50

    # Optimization
    LR_BACKBONE = 1e-4
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_MAX = EPOCHS

    # Loss Function
    SQRT2 = 1.41421356  # Used for Metric-Aligned Laplace Log Likelihood

    # -------------------------------------------------------------------------
    # Inference / Metrics
    # -------------------------------------------------------------------------
    CONFIDENCE_CLIP = 70
    MAX_ERROR_CLIP = 1000

    @classmethod
    def setup(cls):
        """
        Sets up the environment for the experiment.
        1. Creates necessary output directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set fixed random seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Environment setup complete. Working directory: {cls.WORKING_DIR}")
