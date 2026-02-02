import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # File Paths and Directories
    # ==========================================
    # Metadata paths (Input - Read Only)
    TRAIN_METADATA_PATH = "./metadata/train.parquet"
    VAL_METADATA_PATH = "./metadata/val.parquet"
    TEST_METADATA_PATH = "./metadata/test.parquet"
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # Working directory for artifacts (Output - Write Allowed)
    # Using 'idea_7' as per the prompt context
    WORKING_DIR = "./working/idea_7/"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    SEQ_LENGTH = 107
    SEQ_SCORED = 68

    # Target columns provided in training data
    TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    # Columns specifically used for the competition metric
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    # Error columns for weighted loss
    ERROR_COLS = [
        "reactivity_error",
        "deg_error_Mg_pH10",
        "deg_error_pH10",
        "deg_error_Mg_50C",
        "deg_error_50C",
    ]

    # Feature Dimensions
    # 4 bases (A,G,C,U) + 1 (paired/unpaired) + 7 loop types + 1 (paired_base_feature)
    # The exact input dim depends on the feature engineering implementation.
    # Based on the idea: One-Hot Sequence (4) + Structure (3: (, ), .) + Loop (7) = 14
    # Spatially augmented means we concat pair features, so 14 * 2 = 28.
    INPUT_DIM = 28
    OUTPUT_DIM = 5  # Predicting 5 targets

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Convolutional Stem
    CONV_FILTERS = 256
    CONV_KERNEL_SIZE = 3

    # Recurrent Backbone
    RNN_HIDDEN_DIM = 256
    RNN_LAYERS = 2
    RNN_DROPOUT = 0.3
    BIDIRECTIONAL = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0

    # Loss Configuration
    USE_WEIGHTED_LOSS = True
    LOSS_EPSILON = 1e-6  # To prevent division by zero in weights

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR

    # System
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging
    # Set to an integer (e.g., 100) to train on a small subset for testing pipeline
    # Set to None for full training
    DEBUG_SUBSET_SIZE = None

    @staticmethod
    def setup_system():
        """
        Creates necessary directories and sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(Config.SEED)
            torch.cuda.manual_seed_all(Config.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
