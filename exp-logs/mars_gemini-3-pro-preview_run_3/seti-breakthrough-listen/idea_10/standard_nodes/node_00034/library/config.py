import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration module for Idea 10: Siamese EfficientNet-B0 with Adaptive Difference and GeM Pooling.
    """

    # --- Directory and File Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata paths (pre-generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for model checkpoints and caching
    WORK_DIR = "./working/idea_10"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Specific file paths
    MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Parameters ---
    # Original spectrogram dimensions
    ORIG_HEIGHT = 273
    ORIG_WIDTH = 256

    # Input dimensions for the model (padded to nearest multiple of 32)
    # We pad 273 -> 288
    IMG_HEIGHT = 288
    IMG_WIDTH = 256

    # Channels per Siamese branch (On-Target: A, C, E; Off-Target: B, D, F)
    IN_CHANNELS = 3

    # --- Model Architecture ---
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True
    # Initial parameter p for Generalized Mean Pooling
    GEM_P = 3.0

    # --- Training Hyperparameters ---
    SEED = 42
    EPOCHS = 15
    BATCH_SIZE = 32

    # Optimizer settings (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler settings (CosineAnnealingLR)
    T_MAX = 15

    # Augmentation settings
    MIXUP_ALPHA = 0.2

    # --- Inference Parameters ---
    # Number of TTA variations (Original, H-Flip, V-Flip, HV-Flip)
    TTA_STEPS = 4

    # --- System Configuration ---
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Initialize the environment: create directories and set random seeds.
        """
        # Create output directories
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set random seeds for reproducibility
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
