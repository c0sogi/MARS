import os
import torch
import random
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Cache directory for Idea 1 (End-to-End ResNet-18)
    # Required: Ensure this directory exists
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Final Submission Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42

    # Data
    IMAGE_SIZE = (224, 224)  # Standard resolution for ResNet
    NUM_WORKERS = 4  # Number of DataLoader workers

    # Training
    BATCH_SIZE = 128  # Efficient for A100 40GB
    LEARNING_RATE = 1e-4  # Conservative LR for fine-tuning
    EPOCHS = 3  # Short training duration for baseline

    # Model
    MODEL_NAME = "resnet34"
    NUM_CLASSES = 1  # Binary classification (Dog vs Cat)

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------------------------------
    # Setup Utilities
    # -------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)

        # Set seeds
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed):
        """
        Sets seeds for python, numpy, and torch to ensure reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Automatically run setup when config is imported to ensure environment is ready
Config.setup()
