import os
import torch
import numpy as np
import random


class Config:
    # ==============================
    # Path Configuration
    # ==============================
    ROOT_DIR = "."
    INPUT_DIR = os.path.join(ROOT_DIR, "input")
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata paths (pre-generated)
    METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directories for artifacts
    WORKING_DIR = os.path.join(ROOT_DIR, "working", "idea_2")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")

    # Submission output
    SUBMISSION_DIR = os.path.join(ROOT_DIR, "submission")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================
    # Data Configuration
    # ==============================
    # 2.5D Input: Stacking 3 slices (e.g., z, z-1, z-2)
    IN_CHANNELS = 3

    # Input resolution (Height, Width)
    IMG_SIZE = (320, 320)

    # Class definitions
    NUM_CLASSES = 3
    CLASSES = ["large_bowel", "small_bowel", "stomach"]
    CLASS2ID = {c: i for i, c in enumerate(CLASSES)}
    ID2CLASS = {i: c for i, c in enumerate(CLASSES)}

    # Robust Normalization (Percentile Clipping)
    PERCENTILE_MIN = 1.0
    PERCENTILE_MAX = 99.0

    # ==============================
    # Model Configuration
    # ==============================
    BACKBONE = "efficientnet-b1"
    ENCODER_WEIGHTS = "imagenet"

    # ==============================
    # Training Configuration
    # ==============================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    EPOCHS = 15
    BATCH_SIZE = 32

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    MIN_LR = 1e-6
    T_MAX = EPOCHS  # For CosineAnnealingLR

    # Hardware
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==============================
    # Metric / Loss Configuration
    # ==============================
    # Weights for the combined competition metric
    METRIC_DICE_WEIGHT = 0.4
    METRIC_HAUSDORFF_WEIGHT = 0.6

    # ==============================
    # Post-Processing Configuration
    # ==============================
    # Minimum volume (in pixels) to keep a connected component in 3D
    MIN_COMPONENT_VOLUME = 100

    @classmethod
    def setup(cls, verbose=True):
        """
        Initialize the environment:
        1. Create necessary directories for outputs.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.PREDICTION_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.set_seed(cls.SEED)

        if verbose:
            print(f"Configuration initialized.")
            print(f"  Device: {cls.DEVICE}")
            print(f"  Backbone: {cls.BACKBONE}")
            print(f"  Input Size: {cls.IMG_SIZE}")
            print(f"  Batch Size: {cls.BATCH_SIZE}")

    @staticmethod
    def set_seed(seed):
        """Sets the seed for random, numpy, and torch to ensure reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
