import os
import torch
import random
import numpy as np


class Config:
    # =========================
    # Paths & Directories
    # =========================
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    WORKING_DIR = "./working"
    # Specific cache directory for this idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_12")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # =========================
    # Data Preprocessing
    # =========================
    # Windowing (Standard Bone Window)
    WINDOW_LEVEL = 400
    WINDOW_WIDTH = 1800

    # Input Dimensions
    NUM_SLICES = 64  # Number of slices sampled per scan (Bag size)
    IMAGE_SIZE = 256  # H, W dimensions (square)
    IN_CHANNELS = 3  # 2.5D Stacking (z-1, z, z+1)

    # =========================
    # Model Architecture
    # =========================
    MODEL_NAME = "resnet34"
    USE_GROUP_NORM = True
    GROUPS = 32
    NUM_CLASSES = 8  # 7 Vertebrae (C1-C7) + 1 Patient Overall

    # =========================
    # Training Hyperparameters
    # =========================
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    EPOCHS = 10
    NUM_WORKERS = 4
    SEED = 42

    # Scheduler
    T_MAX_MULTIPLIER = 1.5  # T_max = 1.5 * epochs

    # =========================
    # Debugging / Development
    # =========================
    DEBUG = False  # Set True to run on subset
    DEBUG_SAMPLE_SIZE = 10  # Number of samples if debugging

    # =========================
    # Compute
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup(seed=42):
        """
        Sets up the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create cache directory
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Set seeds
        Config.SEED = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
