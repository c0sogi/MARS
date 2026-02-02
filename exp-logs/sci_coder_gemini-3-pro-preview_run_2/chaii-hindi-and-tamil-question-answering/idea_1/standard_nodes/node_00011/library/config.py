import os
import torch
import random
import numpy as np


class Config:
    """
    Centralized configuration for the QA task using DistilBERT.
    Includes file paths, model hyperparameters, and setup utilities.
    """

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # File Paths
    # ==========================================
    # Input data (using metadata splits to prevent leakage)
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/val.csv"
    TEST_DATA_PATH = "./metadata/test.csv"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Output directories
    # Using ./working/idea_1 as the cache/working directory
    OUTPUT_DIR = "./working/idea_1"
    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture & Tokenizer
    # ==========================================
    MODEL_CHECKPOINT = "distilbert-base-multilingual-cased"

    # Sliding Window Parameters
    # Contexts are long (~1600 words), so we must chunk them.
    MAX_LENGTH = 384  # Max tokens per window
    DOC_STRIDE = 128  # Overlap between windows

    # Post-processing constraints
    MAX_ANSWER_LENGTH = 30  # Maximum length of a predicted answer
    N_BEST_SIZE = 20  # Number of logits to consider during decoding

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 16  # Conservative batch size for stability
    LEARNING_RATE = 3e-5  # Standard fine-tuning rate
    WEIGHT_DECAY = 0.01
    EPOCHS = 3  # Increased to 3 based on Lesson 00003
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # Hardware settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # For data loading

    # ==========================================
    # Debugging / Development
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SIZE = 100  # Number of samples to use in debug mode

    @staticmethod
    def setup():
        """
        Initialize the environment:
        1. Create necessary output directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            # Deterministic operations where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Configuration setup complete. Device: {Config.DEVICE}")
