import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for the Regularized Non-Linear Dense-Context Network.
    Includes paths, data dimensions, model hyperparameters, and training settings.
    """

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_26"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (Explicit Cache Invalidation Keys)
    # Using specific version names to ensure fresh feature generation
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data_linear_dense_v1.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data_linear_dense_v1.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data_linear_dense_v1.npz")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    SEQ_LEN = 107
    PRED_LEN = 68

    # Input Feature Dimensions
    # Sequence (4: A,G,U,C)
    # Structure (3: (, ), .)
    # Loop Type (7: S,M,I,B,H,E,X)
    # Partner Identity (4: A,G,U,C) - Explicitly retained
    INPUT_CHANNELS = 4 + 3 + 7 + 4  # Total: 18

    # Targets
    # All 5 targets present in training data and required for submission
    ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Only these 3 are scored and used for masked loss calculation
    SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    NUM_TARGETS = 5

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    # Backbone: Dense Dilated TCN
    GROWTH_RATE = 64  # Increased capacity
    KERNEL_SIZE = 3
    DILATIONS = [1, 2, 4, 8, 16, 32]
    DROPOUT = 0.1  # Regularization

    # Neck: Linear Bottleneck (Cite Lesson 00068)
    # Projects dense history to this dimension before interaction
    LATENT_DIM = 32

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    NUM_WORKERS = 2

    # Scheduler settings
    PATIENCE = 5
    FACTOR = 0.5

    # =========================================================================
    # Debugging / Subsetting
    # =========================================================================
    # Set to an integer (e.g., 100) to train on a small subset for debugging
    SUBSET_SIZE = None

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
