import os
import random
import warnings
import numpy as np
import torch


class Config:
    """
    Configuration class for the Sentiment Extraction task using DeBERTa-v3-Large.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 4
    DEBUG = False  # Set to True to run on a small subset for debugging

    # =========================================================================
    # Model Settings
    # =========================================================================
    MODEL_NAME = "microsoft/deberta-v3-large"
    MAX_LEN = 96

    # =========================================================================
    # Training Settings
    # =========================================================================
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.1
    SCHEDULER = "cosine"
    WARMUP_RATIO = 0.1
    CLIP_GRAD_NORM = 1.0

    # =========================================================================
    # Data Processing Strategy
    # =========================================================================
    FILTER_NEUTRAL = True  # Filter out neutral tweets during training

    # =========================================================================
    # Paths
    # =========================================================================
    # Input Metadata (Pre-generated)
    TRAIN_META_PATH = "./metadata/train.csv"
    VAL_META_PATH = "./metadata/val.csv"
    TEST_META_PATH = "./metadata/test.csv"

    # Artifacts and Caching
    WORKING_DIR = "./working/idea_4/"

    # Submission
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        3. Configures system warnings and flags.
        """
        # 1. Create Directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # 2. Set Random Seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)

        # Ensure deterministic behavior where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 3. System Configuration
        # Suppress specific warnings from transformers/tokenizers
        warnings.filterwarnings("ignore")
        # Disable tokenizer parallelism to prevent deadlocks in DataLoader
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        print(f"Environment setup complete. Working directory: {cls.WORKING_DIR}")
