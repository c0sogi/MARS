import os
import random
import numpy as np
import torch


class Config:
    # --- Project Information ---
    PROJECT_NAME = "SETI_Technosignature_Detection"
    IDEA_NAME = "idea_1"  # Baseline: Spatial Difference CNN
    SEED = 42
    DEBUG = False  # Toggle for debugging with smaller data subsets

    # --- Hardware Settings ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use 4 workers as a safe default for 12 vCPUs to avoid overhead
    NUM_WORKERS = 4

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Directories
    # All artifacts for this idea go here
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    # Cache for preprocessed data (e.g., difference maps)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    # Directory for final submission
    SUBMISSION_DIR = "./submission"

    # File Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "spatial_difference_cnn.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --- Data Parameters ---
    # Raw data: (6 positions, 273 freq bins, 256 time bins)
    RAW_SHAPE = (6, 273, 256)
    # Model input: (1 channel difference map, 273, 256)
    INPUT_SHAPE = (1, 273, 256)

    # --- Training Hyperparameters ---
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    MAX_EPOCHS = 20
    PATIENCE = 3  # Early stopping patience

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # 1. Create Directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # 2. Set Random Seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration setup complete. Device: {cls.DEVICE}")
