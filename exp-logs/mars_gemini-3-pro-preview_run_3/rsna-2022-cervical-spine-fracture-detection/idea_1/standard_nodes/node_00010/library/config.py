import os
import torch
import random
import numpy as np


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_1")
    CHECKPOINT_DIR = IDEA_DIR
    CACHE_DIR = IDEA_DIR  # For caching processed data

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Preprocessing & Augmentation
    # =========================================================================
    IMG_SIZE = 256
    NUM_SLICES = 96  # Uniformly subsample this many slices per scan
    IN_CHANNELS = 3  # 2.5D Stacking: (z-1, z, z+1)

    # Bone Windowing Parameters (Hounsfield Units)
    WINDOW_CENTER = 300
    WINDOW_WIDTH = 2000

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE = "resnet18"
    NUM_CLASSES = 7  # C1, C2, C3, C4, C5, C6, C7
    PRETRAINED = True

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = (
        8  # Number of studies per batch (effective batch size depends on slices)
    )
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 12

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Labels & Targets
    # =========================================================================
    TARGET_COLS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    OVERALL_COL = "patient_overall"

    # =========================================================================
    # Debugging
    # =========================================================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 20  # Number of samples to use when DEBUG is True

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary output directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
