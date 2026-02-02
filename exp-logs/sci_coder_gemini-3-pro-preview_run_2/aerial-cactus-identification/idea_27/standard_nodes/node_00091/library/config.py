import os
import torch


class Config:
    """
    Global configuration for the Cactus Classification pipeline.
    Implements settings for the 'Custom Wide Coordinate-Res2NeXt with Multi-Scale Aggregation' strategy.
    """

    # ==========================================
    # Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Image Root (Metadata contains relative paths from here)
    IMAGE_ROOT = INPUT_DIR

    # Output Directories
    # All outputs for this specific idea iteration go here
    WORKING_DIR = "./working/idea_28"
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Native resolution is 32x32. We strictly avoid resizing.
    IMAGE_SIZE = (32, 32)
    NUM_CLASSES = 1

    # Debugging: Set to an integer (e.g., 100) to train on a small subset.
    # Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Ensemble Strategy: Homogeneous Seed Averaging
    SEEDS = [0, 1, 2, 3, 4]

    NUM_EPOCHS = 20
    BATCH_SIZE = 128

    # Optimizer (AdamW) & Scheduler (Cosine Annealing) settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    ETA_MIN = 1e-6  # Minimum LR for Cosine Annealing

    # ==========================================
    # Hardware Configuration
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available, 4 workers is a safe balance
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
