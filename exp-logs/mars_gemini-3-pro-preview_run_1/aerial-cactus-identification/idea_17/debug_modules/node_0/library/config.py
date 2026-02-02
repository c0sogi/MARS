import os
import torch
import numpy as np
import random


class Config:
    # --------------------------------------------------------------------------
    # Project & System Settings
    # --------------------------------------------------------------------------
    PROJECT_NAME = "idea_17"
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Available resources: 12 vCPUs, 220 GB RAM, A100 GPU
    NUM_WORKERS = 4

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Base Directories
    BASE_DIR = os.getcwd()
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    METADATA_DIR = os.path.join(BASE_DIR, "metadata")

    # Input Data
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Original Full Train CSV (Required for 5-Fold CV)
    FULL_TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories (Idea 17 specific)
    WORKING_DIR = os.path.join(BASE_DIR, "working", PROJECT_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # --------------------------------------------------------------------------
    # Data Configuration
    # --------------------------------------------------------------------------
    IMG_SIZE = 32
    NUM_CLASSES = 1

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    NUM_FOLDS = 5
    EPOCHS = 30
    BATCH_SIZE = 128

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Regularization
    MIXUP_ALPHA = 0.2

    # Multi-Task Learning (Auxiliary Quality Head)
    # Weight for the Mean Squared Error of the log(file_size) prediction
    MTL_WEIGHT = 0.1

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 25
    SWA_LR = 1e-4

    # --------------------------------------------------------------------------
    # Model Architecture Configuration
    # --------------------------------------------------------------------------
    # The ensemble consists of these three diverse backbones
    BACKBONES = ["RepVGG", "ResNet", "NeXt"]

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary directories.
        2. Set random seeds for full reproducibility.
        """
        # Create directories
        for d in [
            cls.WORKING_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.SUBMISSION_DIR,
        ]:
            os.makedirs(d, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Execute setup on import to ensure environment is ready
Config.setup()
