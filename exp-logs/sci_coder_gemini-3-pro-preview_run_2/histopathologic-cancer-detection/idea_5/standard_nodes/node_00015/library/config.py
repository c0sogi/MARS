import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the pathology tumor detection task.
    Implements the settings for a 5-Fold Ensemble of GeM-Pooled ConvNeXt-Tiny models.
    """

    # --- General Configuration ---
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for debugging
    DEBUG_DATA_LIMIT = 1000  # Number of samples to use when DEBUG is True

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Configuration ---
    IMAGE_SIZE = 96  # Original input patch size
    CROP_SIZE = 64  # Center crop size for training and inference
    NUM_CLASSES = 1  # Binary classification (Tumor vs Non-Tumor)

    # Dataset Statistics (Calculated from EDA)
    # Used for normalization instead of ImageNet defaults
    DATASET_MEAN = [0.7035, 0.5476, 0.6975]
    DATASET_STD = [0.2388, 0.2821, 0.2159]

    # --- Model Configuration ---
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True
    USE_GEM_POOLING = True
    GEM_P_INIT = 3.0
    # Critical: Retain LayerNorm before GeM pooling to ensure stability
    LAYERNORM_BEFORE_POOLING = True
    DROP_PATH_RATE = 0.1

    # --- Training Configuration ---
    N_FOLDS = 1
    EPOCHS = 20
    BATCH_SIZE = 128  # Optimized for A100 GPU
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.05
    MIN_LR = 1e-6
    WARMUP_EPOCHS = 2

    # Regularization
    MIXUP_ALPHA = 0.2  # Mixup regularization strength

    # --- Inference Configuration ---
    TTA_VIEWS = 8  # 8-view Test Time Augmentation (Dihedral)

    # --- Hardware ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary working directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        dirs = [cls.WORKING_DIR, cls.CHECKPOINT_DIR, cls.CACHE_DIR, cls.SUBMISSION_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
