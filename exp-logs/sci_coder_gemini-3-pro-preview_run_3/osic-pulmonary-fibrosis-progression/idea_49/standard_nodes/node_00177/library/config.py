import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration for the SCAR-Net (Standardized Constraint-Aware Residual Network) experiment.
    """

    # =========================================================================
    # Reproducibility
    # =========================================================================
    SEED = 42

    def set_seed(self):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Experiment specific directory
    IDEA_NAME = "idea_49"
    IDEA_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(IDEA_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(IDEA_DIR, "submission")

    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Processing & Normalization
    # =========================================================================
    # Image Preprocessing
    IMAGE_SIZE = 260
    NUM_SLICES = 3  # Anchor + 2 boundaries (Top/Bottom 50% area)

    # Radiological Windowing (Lung Window)
    WINDOW_LEVEL = -600
    WINDOW_WIDTH = 1500

    # Global Target Statistics (from EDA) for Standardization
    # Mean: 2654.6528, Std: 801.7017
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Metric Constraint
    # The metric clips sigma at 70ml.
    # We enforce this floor in the standardized space: 70 / TARGET_STD
    SIGMA_FLOOR_STD = 70.0 / TARGET_STD

    # Feature Engineering
    # Scale relative time (Weeks) by 0.01 instead of Z-score
    TIME_SCALE = 0.01

    # =========================================================================
    # Model Architecture (SCAR-Net)
    # =========================================================================
    BACKBONE_NAME = "efficientnet_b2"
    BACKBONE_PRETRAINED = True

    # Dimensions
    BACKBONE_OUT_DIM = 1408  # EfficientNet-B2 final features
    PROJECTION_DIM = 64  # Bottleneck projection for image features
    HIDDEN_DIM = 128  # MLP hidden dimension
    OUTPUT_DIM = 2  # Mean, LogStd (or RawStd)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 30
    NUM_WORKERS = 4

    # Optimization
    # Differential Learning Rates
    BACKBONE_LR = 1e-4
    HEAD_LR = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_MAX = EPOCHS  # For Cosine Annealing
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 8

    # Compute
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Debugging
    # =========================================================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use if DEBUG is True

    @classmethod
    def print_config(cls):
        print("=" * 40)
        print(f"CONFIG: {cls.IDEA_NAME}")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"Image Size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Target Mean: {cls.TARGET_MEAN:.4f}")
        print(f"Target Std: {cls.TARGET_STD:.4f}")
        print(f"Sigma Floor (Std Space): {cls.SIGMA_FLOOR_STD:.6f}")
        print("=" * 40)
