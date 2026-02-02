import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for the Origin-Corrected Parametric Network (OCP-Net).
    Centralizes hyperparameters, paths, and constants.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output paths
    WORKING_DIR = "./working/idea_9"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # Data Hyperparameters
    # =========================================================================
    # Image processing
    IMG_SIZE = 224  # Native resolution for EfficientNet-B0
    NUM_SLICES = 3  # Apical, Middle, Basal

    # Normalization Constants (Derived from EDA)
    # Train FVC Mean: 2654.6528, Std: 801.7017
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Time Engineering
    # We scale the relative time (Weeks - Baseline_Week) by this factor
    # to keep inputs in a neural-network friendly range (approx -1 to 1)
    TIME_SCALE = 100.0

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "ocp_net"
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True

    # Feature Extraction
    PROJECTION_DIM = 128  # Dimension to project flattened image features to

    # Tabular Features
    # Baseline FVC, Age, Sex, SmokingStatus
    N_TABULAR_FEATURES = 4

    # Parametric Heads
    # Trajectory Head: alpha (AR), beta (offset), gamma (slope) -> 3 outputs
    # Uncertainty Head: delta_base, delta_growth -> 2 outputs

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCHS = 50
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    # Differential Learning Rates
    LR_BACKBONE = 1e-4  # Slower learning for pre-trained weights
    LR_HEADS = 1e-3  # Faster learning for new layers
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10

    # Loss Constraints
    MIN_UNCERTAINTY = 0.05  # Numerical stability floor (before post-processing clip)

    # =========================================================================
    # Inference / Post-processing
    # =========================================================================
    SUBMISSION_STD_CLIP = 70.0  # The metric clips at 70, so we clip predictions too

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_transforms(cls, phase="train"):
        """
        Returns the image transformations for the specified phase.
        Uses albumentations (assumed to be imported in the dataset module,
        but config defines parameters).
        """
        # Note: Actual transform object creation is delegated to the dataset module
        # to avoid heavy imports in config. This method serves as a parameter holder.
        if phase == "train":
            return {
                "horizontal_flip_prob": 0.5,
                "rotate_limit": 15,
                "brightness_contrast_prob": 0.2,
                "size": cls.IMG_SIZE,
            }
        else:
            return {"size": cls.IMG_SIZE}
