import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for Siamese Multi-View SegFormer (Siamese-MVS).
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # =========================================================================
    # Directories and File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output and Caching Directory (Idea 13)
    WORKING_DIR = "./working/idea_13"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model Checkpoint
    CHECKPOINT_FILENAME = "best_model.pth"
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, CHECKPOINT_FILENAME)

    # Submission Output
    SUBMISSION_FILENAME = "submission.csv"
    SUBMISSION_PATH = SUBMISSION_FILENAME  # Expected in root/home directory

    # =========================================================================
    # Data Configuration
    # =========================================================================
    TILE_SIZE = 512
    STRIDE = 512

    # Siamese Multi-View Topology
    # Defines the Z-slice ranges for the three parallel input streams.
    # Each view covers a 24-slice range, processed into 3 channels (Overlapping Thick Slab).
    SLICES_VIEWS = {
        "high": (16, 40),  # View 1: Upper range
        "center": (20, 44),  # View 2: Center range
        "low": (24, 48),  # View 3: Lower range
    }

    # Input dimensions
    CHANNELS_PER_VIEW = 3  # RGB-like input for ImageNet pretrained weights

    # =========================================================================
    # Model Architecture
    # =========================================================================
    ENCODER_NAME = "segformer_mit_b2"
    ENCODER_WEIGHTS = "imagenet"
    NUM_CLASSES = 1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42

    # Batch size adjusted for Siamese network memory footprint (3x encoder passes)
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 2e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 15

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # Validation Logic Gate
    # Submission is only generated if Validation F0.5 Score > BASELINE_SCORE
    BASELINE_SCORE = 0.598

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Debugging
    # =========================================================================
    DEBUG = False
    MAX_DEBUG_SAMPLES = 100  # Restrict dataset size when DEBUG is True

    @staticmethod
    def set_seed(seed=42):
        """
        Sets fixed random seeds for reproducibility across libraries.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
