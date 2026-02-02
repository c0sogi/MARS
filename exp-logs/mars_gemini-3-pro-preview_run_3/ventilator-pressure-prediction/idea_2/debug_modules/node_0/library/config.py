import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration for the Ventilator Pressure Prediction pipeline.
    Acts as the single source of truth for paths, hyperparameters, and feature definitions
    to ensure synchronization between data processing and model architecture.
    """

    # ==========================================
    # Paths
    # ==========================================
    # Input Metadata (Pre-split using GroupKFold strategy)
    TRAIN_DATA_PATH = "./metadata/train.csv"
    VAL_DATA_PATH = "./metadata/validation.csv"
    TEST_DATA_PATH = "./metadata/test.csv"

    # Output Directories
    WORKING_DIR = "./working/idea_2/"
    SUBMISSION_DIR = "./submission/"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Cache files for processed tensors
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_cache.npy")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_cache.npy")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_cache.npy")

    # ==========================================
    # Feature Engineering Configuration
    # ==========================================
    # Raw input columns from dataset
    RAW_COLS = ["time_step", "u_in", "u_out", "R", "C"]

    # Features to be generated and used by the model.
    # The order here dictates the channel order in the input tensor.
    FEATURE_COLS = [
        "time_step",  # Timestamp
        "u_in",  # Control input (inspiratory)
        "u_out",  # Control input (expiratory)
        "R",  # Resistance
        "C",  # Compliance
        "u_in_cumsum",  # Integral term (Volume proxy)
        "u_in_diff1",  # 1st Derivative (Velocity)
        "u_in_diff2",  # 2nd Derivative (Acceleration)
        "R_flow",  # Interaction: R * u_in
        "C_volume",  # Interaction: u_in_cumsum / C
    ]

    # Target variable
    TARGET_COL = "pressure"

    # Identifiers
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"

    # Sequence Length (Fixed for this dataset)
    SEQ_LEN = 80

    # Input dimension for the model (calculated automatically)
    INPUT_DIM = len(FEATURE_COLS)

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Hybrid CNN-LSTM Architecture
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 3

    LSTM_HIDDEN_DIM = 512
    LSTM_LAYERS = 4
    LSTM_BIDIRECTIONAL = True
    LSTM_DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    EPOCHS = 100
    BATCH_SIZE = 512  # A100 can handle large batches
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler settings (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000  # Number of breaths to use if DEBUG is True

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def initialize(cls):
        """
        Sets up the environment:
        1. Creates necessary directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Deterministic algorithms
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_feature_indices(cls):
        """Returns a dictionary mapping feature names to their index in the input tensor."""
        return {name: i for i, name in enumerate(cls.FEATURE_COLS)}
