import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories for Idea 6
    WORKING_DIR = "./working/idea_6"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Original patch size
    FULL_IMAGE_SIZE = 96
    # Size after center crop (Contextual Crop strategy)
    INPUT_SIZE = 64
    # Number of workers for data loading
    NUM_WORKERS = 8
    # Dataset specific statistics (from EDA)
    MEAN = [0.7035, 0.5476, 0.6975]
    STD = [0.2388, 0.2821, 0.2159]

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_NAME = "convnext_tiny.fb_in1k"
    NUM_CLASSES = 1
    # Stochastic Depth rate
    DROP_PATH_RATE = 0.2
    # Exponential Moving Average
    USE_EMA = True
    EMA_DECAY = 0.9999

    # =========================================================================
    # Training Configuration
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    EPOCHS = 25
    BATCH_SIZE = 256

    # Optimizer (AdamW)
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 0.05
    MIN_LR = 1e-6

    # Regularization (Mixup)
    MIXUP_ALPHA = 0.2
    MIXUP_PROB = 1.0

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    # Test Time Augmentation steps (8 = Dihedral)
    TTA_STEPS = 8

    # =========================================================================
    # Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def seed_everything(cls, seed=None):
        """
        Sets seeds for reproducibility.
        """
        if seed is None:
            seed = cls.SEED

        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
