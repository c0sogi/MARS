import os
import random
import numpy as np
import torch


class Config:
    """
    Centralized configuration for the pathology tumor detection task.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output directories (Writeable)
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Output files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Original image size is 96x96
    # We crop the center 48x48 to cover the 32x32 target region with context
    CROP_SIZE = 48

    # Normalization parameters (if needed beyond /255.0)
    # Using standard ImageNet means/stds is common, but for custom shallow CNN
    # and pathology data, simple scaling [0, 1] is often sufficient.

    # Debugging / Development
    # Set to a small integer (e.g., 1000) to limit dataset size for rapid testing
    # Set to None for full training
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Model & Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 3

    # ==========================================
    # Compute Settings
    # ==========================================
    # 12 vCPUs available, leaving some overhead
    NUM_WORKERS = 8

    # Automatic device selection
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_seed(seed: int = 42):
        """
        Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
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

    @staticmethod
    def setup_directories():
        """
        Creates necessary output directories if they don't exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
