import os
import random
import numpy as np
import torch


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for Idea 8 (CoConvNeXt-UNet)
    WORKING_DIR = "./working/idea_8"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "coconvnext_unet_best.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    PATCH_SIZE = 128
    # High Density Sampling: 100 patches per image per epoch
    PATCHES_PER_IMAGE = 100
    NUM_WORKERS = 4

    # =========================================================================
    # Model Parameters
    # =========================================================================
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    BASE_FILTERS = 64

    # =========================================================================
    # Training Parameters
    # =========================================================================
    SEED = 42
    NUM_EPOCHS = 100
    BATCH_SIZE = 32  # Safe for A100 with complex 7x7 kernels and attention

    # Optimizer & Scheduler
    LEARNING_RATE = 1e-4
    MIN_LEARNING_RATE = 1e-6
    WEIGHT_DECAY = 1e-2  # Strong regularization

    EARLY_STOPPING_PATIENCE = 15

    # =========================================================================
    # Inference Parameters
    # =========================================================================
    OVERLAP_RATIO = 0.5
    TTA_ENABLED = True

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
