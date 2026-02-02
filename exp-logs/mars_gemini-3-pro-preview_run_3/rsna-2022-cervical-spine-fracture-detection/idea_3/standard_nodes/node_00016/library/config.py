import os
import torch
import random
import numpy as np


class Config:
    # =========================
    # General Settings
    # =========================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 10
    NUM_WORKERS = 4

    # =========================
    # File Paths
    # =========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Image Directories
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Cache Directory (Idea 3 specific)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_3")

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================
    # Data Preprocessing
    # =========================
    IMAGE_SIZE = (256, 256)

    # Bone Windowing
    WINDOW_CENTER = 300
    WINDOW_WIDTH = 2000

    # MIL Bag Settings
    BAG_SIZE = 64  # Number of slices sampled per study
    IN_CHANNELS = 3  # 2.5D Stacking (z-1, z, z+1)

    # =========================
    # Model Hyperparameters
    # =========================
    BACKBONE = "resnet18"
    NUM_CLASSES = 7  # C1-C7
    PRETRAINED = True
    DROPOUT = 0.0  # No dropout in head as per plan

    # =========================
    # Training Settings
    # =========================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    EPOCHS = 10
    BATCH_SIZE = 8  # Effective batch size = BATCH_SIZE * BAG_SIZE images
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5

    # Scheduler
    T_MAX_MULTIPLIER = 1.5  # T_max = EPOCHS * 1.5

    # Early Stopping
    PATIENCE = 3
    MIN_DELTA = 1e-4

    @classmethod
    def setup(cls, seed=None):
        """
        Initializes the environment:
        1. Sets random seeds for reproducibility.
        2. Creates necessary directories.
        """
        if seed is None:
            seed = cls.SEED

        # 1. Set Random Seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # 2. Create Directories
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        print(f"Configuration setup complete. Device: {cls.DEVICE}, Seed: {seed}")
        print(f"Cache directory: {cls.CACHE_DIR}")
