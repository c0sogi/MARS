import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 17 (Dual-Stream Large-Kernel U-Net)
    WORKING_DIR = "./working/idea_17"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VALID_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Data Configuration
    # ==========================================
    IMAGE_SIZE = 256

    # Temporal sequence details
    N_TIMES_BEFORE = 4
    N_TIMES_AFTER = 3

    # Input Engineering
    # Stream A: Ash False Color Composite (Bands 11, 14, 15)
    ASH_BANDS = [11, 14, 15]

    # Stream B: Raw Band Differences (Bands 11, 14, 15 at t=4 minus t=3)
    DIFF_BANDS = [11, 14, 15]

    # Total input channels logically (3 for Ash + 3 for Diff)
    IN_CHANNELS_STREAM_A = 3
    IN_CHANNELS_STREAM_B = 3

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "DualStreamUNet"
    BACKBONE = "convnext_tiny"
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 30

    # Optimizer settings
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler settings
    T_MAX = 30  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLES = 1000  # Subset size when DEBUG is True

    # ==========================================
    # Inference Configuration
    # ==========================================
    THRESHOLD = 0.5
    USE_TTA = True  # Test Time Augmentation (Flip/Rotate)

    @classmethod
    def setup(cls):
        """
        Initialize the environment: create necessary directories and set random seeds.
        """
        # Ensure working and submission directories exist
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducibility seeds
        cls.set_seed()

    @classmethod
    def set_seed(cls):
        """
        Set fixed random seeds for reproducibility across libraries.
        """
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Enforce deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
