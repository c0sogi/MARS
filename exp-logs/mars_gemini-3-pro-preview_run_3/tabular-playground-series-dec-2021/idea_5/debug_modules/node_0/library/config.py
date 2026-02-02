import os
import torch


class Config:
    # ==========================================
    # Project & Paths
    # ==========================================
    PROJECT_NAME = "idea_5"

    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Paths (Parquet files)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output Directories
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"

    # Processed Data Cache Paths
    # We use .npy for processed arrays to ensure fast loading
    TRAIN_X_PATH = os.path.join(WORKING_DIR, "train_X.npy")
    TRAIN_Y_PATH = os.path.join(WORKING_DIR, "train_y.npy")
    VAL_X_PATH = os.path.join(WORKING_DIR, "val_X.npy")
    VAL_Y_PATH = os.path.join(WORKING_DIR, "val_y.npy")
    TEST_X_PATH = os.path.join(WORKING_DIR, "test_X.npy")
    TEST_IDS_PATH = os.path.join(WORKING_DIR, "test_ids.npy")

    # Model & Submission Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "grn_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Target Column
    TARGET_COL = "Cover_Type"
    ID_COL = "Id"

    # Number of classes (Cover_Type 1-7, mapped to 0-6 internally)
    NUM_CLASSES = 7

    # Feature Engineering Flags
    USE_GEOMETRIC_FEATURES = True
    NORMALIZE_FEATURES = True

    # Debugging
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50000  # Number of rows to use if DEBUG is True

    # ==========================================
    # Model Architecture (GRN)
    # ==========================================
    # Input dimension will be calculated dynamically based on feature engineering
    HIDDEN_DIM = 512
    NUM_LAYERS = 3  # Number of Gated Residual Blocks
    DROPOUT = 0.2
    USE_BATCH_NORM = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 2048  # Large batch size for tabular data on A100
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Regularization

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 3
    SCHEDULER_MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 7

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    @classmethod
    def setup(cls):
        """
        Creates necessary output directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        import numpy as np
        import random

        torch.manual_seed(cls.SEED)
        torch.cuda.manual_seed_all(cls.SEED)
        np.random.seed(cls.SEED)
        random.seed(cls.SEED)

        # Ensure deterministic behavior for cudnn if needed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
