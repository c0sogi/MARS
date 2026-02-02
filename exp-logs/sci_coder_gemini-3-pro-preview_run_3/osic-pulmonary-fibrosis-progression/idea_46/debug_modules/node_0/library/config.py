import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Boundary-Constrained Output-Space Residual Network.
    Centralizes all file paths, hyperparameters, and constants.
    """

    # ==========================
    # Path Configuration
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for this specific idea/experiment
    IDEA_NAME = "idea_46"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINTS_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # ==========================
    # Data Configuration
    # ==========================
    IMG_SIZE = 260
    NUM_SLICES = 3  # Anchor slice + 2 boundary slices (50% area)

    # Normalization Statistics (Derived from EDA)
    # Target Variable: FVC
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Input Scalars (for standardization where applicable)
    AGE_MEAN = 67.5825
    AGE_STD = 6.6259

    # ==========================
    # Model Configuration
    # ==========================
    BACKBONE = "efficientnet_b2"
    PRETRAINED = True

    # Dimensions
    PROJECTION_DIM = 64  # Bottleneck for image features
    HIDDEN_DIM = 128  # MLP hidden dimension
    OUTPUT_DIM = 2  # Mean and Std

    # Regularization & Constraints
    DROP_RATE = 0.2
    SIGMA_FLOOR = 70.0  # Architecturally enforced minimum uncertainty

    # ==========================
    # Training Configuration
    # ==========================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 50

    # Optimization
    LR_BACKBONE = 1e-4  # Lower learning rate for the pre-trained backbone
    LR_HEAD = 1e-3  # Higher learning rate for the MLP heads
    WEIGHT_DECAY = 1e-2
    T_MAX = EPOCHS  # For Cosine Annealing scheduler

    # Hardware & Runtime
    NUM_WORKERS = 4  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging
    DEBUG = False  # Set to True to run on a subset
    N_DEBUG_SAMPLES = 100  # Number of samples to use when DEBUG is True

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary working directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINTS_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.set_seed()

        # Print configuration status
        print(f"Configuration initialized for {cls.IDEA_NAME}")
        print(f"Device: {cls.DEVICE}")
        print(f"Artifacts will be stored in: {cls.WORKING_DIR}")

    @classmethod
    def set_seed(cls):
        """Sets fixed random seeds for reproducibility."""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior for cuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
