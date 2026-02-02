import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Siamese Dual-Encoder pipeline.
    Handles hyperparameters, file paths, and environment setup.
    """

    # ==========================
    # General Settings
    # ==========================
    SEED = 42
    DEBUG = False  # Set to True to train on a small subset for debugging
    DEBUG_SUBSET_SIZE = 1000  # Number of samples to use when DEBUG is True

    # ==========================
    # Data Paths
    # ==========================
    # Metadata paths (Input)
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"

    # Working directory for artifacts (cache, checkpoints)
    WORKING_DIR = "./working/idea_2/"

    # Model checkpoint path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission paths
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================
    # Model Hyperparameters
    # ==========================
    # Using a lightweight DeBERTa model as the backbone
    MODEL_NAME = "microsoft/deberta-v3-xsmall"

    # Max length 512 to accommodate [CLS] Prompt [SEP] Response [SEP]
    MAX_LENGTH = 512

    # Target classes: Winner Model A, Winner Model B, Tie
    NUM_CLASSES = 3

    # ==========================
    # Training Hyperparameters
    # ==========================
    EPOCHS = 3
    LEARNING_RATE = 2e-5  # Lower learning rate for fine-tuning
    TRAIN_BATCH_SIZE = 8  # Reduced to fit 16GB VRAM with Siamese architecture
    VALID_BATCH_SIZE = 16

    # Regularization
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.1

    # Optimization
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    EARLY_STOPPING_PATIENCE = 2

    # ==========================
    # Compute Settings
    # ==========================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    PIN_MEMORY = True

    @staticmethod
    def setup():
        """
        Prepares the environment for execution:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create working and submission directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds for reproducibility
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(
            f"Environment setup complete. Working dir: {Config.WORKING_DIR}, Device: {Config.DEVICE}"
        )
