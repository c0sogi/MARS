import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate data and saving models
    # Using 'idea_3' as the current iteration workspace
    WORKING_DIR = "./working/idea_3"

    # Directory for final submission
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    MODEL_PATH = os.path.join(WORKING_DIR, "resunet_plusplus_best.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    PATCH_SIZE = 128
    PATCHES_PER_IMAGE = 100  # High density sampling as per strategy
    NUM_WORKERS = 8  # Utilizing available vCPUs
    PIN_MEMORY = True

    # Caching
    LOAD_CACHED_DATA = True  # If True, tries to load .npy files from WORKING_DIR

    # =========================================================================
    # Model Parameters (Deeply Supervised ResUNet++)
    # =========================================================================
    IN_CHANNELS = 1  # Grayscale input
    OUT_CHANNELS = 1  # Single channel output (Noise Residual)
    BASE_CHANNELS = 32  # Base filter count
    DEEP_SUPERVISION = True  # Enable multi-head output for UNet++

    # =========================================================================
    # Training Parameters
    # =========================================================================
    BATCH_SIZE = 64  # A100 40GB can handle large batches
    EPOCHS = 100  # Max epochs, controlled by early stopping
    LEARNING_RATE = 1e-3  # AdamW initial learning rate
    WEIGHT_DECAY = 1e-2  # Aggressive regularization

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15
    MIN_DELTA = 1e-6

    # =========================================================================
    # Inference Parameters
    # =========================================================================
    TILE_SIZE = 512  # Larger tile size for inference to capture context
    TILE_OVERLAP = 0.5  # 50% overlap to reduce boundary artifacts
    TTA_ENABLED = True  # Test Time Augmentation (Flips/Rotates)

    # =========================================================================
    # Compute & Reproducibility
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Debugging
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLES = 10  # Number of images to use in debug mode

    @staticmethod
    def setup():
        """
        Ensures necessary directories exist and sets random seeds.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        Config.seed_everything(Config.SEED)

    @staticmethod
    def seed_everything(seed: int):
        """
        Sets the seed for reproducibility across random, numpy, and torch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Automatically setup environment on import
Config.setup()
