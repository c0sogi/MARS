import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration for the Deeply Supervised ResUNet++ Denoising Task.
    Centralizes all hyperparameters, file paths, and setup logic.
    """

    # -------------------------------------------------------------------------
    # Reproducibility & Hardware
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Utilizing 12 vCPUs; leaving some buffer
    NUM_WORKERS = 8

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

    # Outputs
    # Cache directory for storing processed/patched data
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    # Model checkpoint path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "resunet_plusplus_best.pth")
    # Final submission file
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    # High-Density Patching Strategy
    PATCH_SIZE = 64
    PATCHES_PER_IMAGE = 100  # Extract 100 patches per image per epoch

    # Normalization (Input is 0-1, so we generally keep it simple or standardize)
    # Here we assume 0-1 input range.

    # -------------------------------------------------------------------------
    # Model Architecture: ResUNet++
    # -------------------------------------------------------------------------
    MODEL_NAME = "ResUNetPlusPlus"
    BASE_FILTERS = 64  # High capacity retention
    DEEP_SUPERVISION = True  # Weighted loss from nested decoder branches
    USE_SE = True  # Squeeze-and-Excitation blocks
    USE_SILU = True  # SiLU activation

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    NUM_EPOCHS = 100
    # A100 40GB allows for decent batch size with 64x64 patches
    BATCH_SIZE = 64

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2  # Aggressive regularization

    # Scheduler: Cosine Annealing
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 15

    # -------------------------------------------------------------------------
    # Inference Configuration
    # -------------------------------------------------------------------------
    # Tiled Inference to prevent boundary artifacts
    TILE_SIZE = 256
    TILE_OVERLAP = 64

    # Test Time Augmentation (Flip/Rotate averaging)
    USE_TTA = True

    @staticmethod
    def initialize():
        """
        Sets up the environment:
        1. Creates necessary working and submission directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set Seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration initialized. Device: {Config.DEVICE}")
        print(f"Working Directory: {Config.WORKING_DIR}")
