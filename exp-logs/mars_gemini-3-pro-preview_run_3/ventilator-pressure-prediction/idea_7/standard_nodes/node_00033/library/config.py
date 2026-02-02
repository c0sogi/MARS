import os
import random
import numpy as np
import torch


class Config:
    """
    Centralized configuration for the Ventilator Pressure Prediction task.
    Implements the settings for the Fully Contextualized Pyramidal Hybrid (FCP-Net).
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Input Metadata Files (Pre-split)
    TRAIN_FILE = os.path.join(METADATA_DIR, "train.csv")
    VAL_FILE = os.path.join(METADATA_DIR, "validation.csv")
    TEST_FILE = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Cache Files (Parquet for efficiency)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test.parquet")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    SEED = 42

    # Training
    BATCH_SIZE = 128  # Strict requirement for gradient noise regularization
    EPOCHS = 60  # Range 60-80 for convergence
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    PATIENCE = 15  # For Early Stopping
    FACTOR = 0.5  # For ReduceLROnPlateau
    MIN_LR = 1e-6

    # Architecture (FCP-Net)
    TCN_DIM = 64  # Base dimension for TCN
    LSTM_DIM = (
        256  # High capacity for Integral states (Cite solution_lesson_node_00032)
    )
    LSTM_LAYERS = 3  # Deep Bidirectional LSTM
    KERNEL_SIZE = 5  # Wide kernel (5 or 7) for TCN
    DROPOUT = 0.1

    # ==========================================
    # Data & Feature Engineering
    # ==========================================
    SEQ_LEN = 80  # Fixed breath length

    # Feature Columns
    # Single source of truth to prevent generation-selection mismatch.
    # Includes PID states and Physics interactions.
    FEATURE_COLS = [
        "time_step",  # Time
        "u_in",  # Control Input (Proportional)
        "u_out",  # Expiratory Valve (State)
        "R",  # Resistance (Static)
        "C",  # Compliance (Static)
        "u_in_cumsum",  # Integral (Volume proxy)
        "u_in_diff1",  # Derivative (Flow accel proxy)
        "u_in_diff2",  # Second Derivative (Jerk proxy)
        "R_u_in",  # Physics: Interaction of R and Flow
        "vol_C",  # Physics: Interaction of Volume and Compliance
    ]

    # Target Column
    TARGET_COL = "pressure"

    # Derived property
    INPUT_DIM = len(FEATURE_COLS)

    @classmethod
    def setup_directories(cls, clean_cache=True):
        """
        Creates necessary directories and optionally cleans old cache files
        to ensure training on fresh data.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        if clean_cache:
            print(f"Cleaning cache in {cls.WORKING_DIR}...")
            extensions = [".npy", ".pt", ".parquet", ".pth"]
            for f in os.listdir(cls.WORKING_DIR):
                if any(f.endswith(ext) for ext in extensions):
                    try:
                        os.remove(os.path.join(cls.WORKING_DIR, f))
                        print(f"Deleted: {f}")
                    except OSError as e:
                        print(f"Error deleting {f}: {e}")


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When using CuDNN backend, two options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed environment variable for hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")
