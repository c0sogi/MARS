import os
import torch
import random
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    PROJECT_NAME = "idea_20"
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEBUG_SAMPLES = 5000  # Number of samples to use if DEBUG is True

    # -------------------------------------------------------------------------
    # Compute Environment
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs for data loading, default to 4 if detection fails
    NUM_WORKERS = 8

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"

    # Input Data Paths (using metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Original Raw Data (for reference/vocab building)
    ORIGINAL_TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    ORIGINAL_TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths
    TRAIN_PROCESSED_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_PROCESSED_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_PROCESSED_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")
    METADATA_CACHE_PATH = os.path.join(WORKING_DIR, "metadata.npy")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Definitions
    # -------------------------------------------------------------------------
    ID_COL = "id"
    TARGET_COL = "target"

    # Continuous features: f_00 to f_26, f_28 (f_27 is categorical string)
    # Plus derived feature: unique_character_count
    CONTINUOUS_FEATURES = [f"f_{i:02d}" for i in range(29) if i != 27] + [
        "unique_character_count"
    ]

    # Categorical features: f_29, f_30 are discrete
    # Plus decomposed f_27 characters (10 positions)
    CATEGORICAL_FEATURES = ["f_29", "f_30"] + [f"f_27_char_{i}" for i in range(10)]

    # -------------------------------------------------------------------------
    # Model Architecture: Heterogeneous Parallel Funnel Ensemble (HPFE)
    # -------------------------------------------------------------------------
    EMBEDDING_DIM = 16  # Dimension for all categorical embeddings

    # Stream Configurations
    # 5 Independent Streams with varying capacity and regularization
    STREAMS_CONFIG = [
        # Stream 1: Anchor (Standard Funnel)
        {"hidden_dims": [512, 256, 128], "dropout": 0.20},
        # Stream 2: Anchor (Standard Funnel)
        {"hidden_dims": [512, 256, 128], "dropout": 0.20},
        # Stream 3: High Capacity (Wide Funnel)
        {"hidden_dims": [1024, 512, 256], "dropout": 0.25},
        # Stream 4: Aggressive Fit (Low Dropout)
        {"hidden_dims": [512, 256, 128], "dropout": 0.15},
        # Stream 5: Conservative Fit (High Dropout)
        {"hidden_dims": [512, 256, 128], "dropout": 0.25},
    ]

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 1024
    MAX_EPOCHS = 50
    PATIENCE = 7  # Early stopping patience

    # Optimization
    LEARNING_RATE = 1e-3  # Max LR for OneCycleLR
    WEIGHT_DECAY = 1e-5  # Optimal for Adam in this context

    @staticmethod
    def set_seed(seed=None):
        """Sets the random seed for reproducibility."""
        if seed is None:
            seed = Config.SEED

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensure deterministic behavior where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
