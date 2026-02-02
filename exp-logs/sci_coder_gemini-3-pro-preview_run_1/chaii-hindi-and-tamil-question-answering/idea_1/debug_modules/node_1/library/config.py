import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Hindi/Tamil Question Answering task.
    """

    # -------------------------------------------------------------------------
    # Reproducibility & General
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output directories (Writeable)
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
    SUBMISSION_DIR = "./submission"

    # File Paths
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model")
    TOKENIZER_SAVE_PATH = os.path.join(CACHE_DIR, "tokenizer")

    # -------------------------------------------------------------------------
    # Model Architecture & Tokenization
    # -------------------------------------------------------------------------
    MODEL_NAME = "google/mt5-small"

    # Contexts are long (mean ~1600 words), so we use a larger source length.
    # A100 40GB can handle 1024-1280 tokens with mt5-small easily.
    MAX_SOURCE_LENGTH = 1024
    MAX_TARGET_LENGTH = 64

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 15
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01
    PATIENCE = 3  # Early stopping patience
    GRADIENT_ACCUMULATION_STEPS = 1

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

    @classmethod
    def setup(cls):
        """
        Initializes the environment: creates directories and sets seeds.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.set_seed(cls.SEED)

        # Print device info
        if cls.DEBUG:
            print(f"Config Setup Complete. Device: {cls.DEVICE}")
