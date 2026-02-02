import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    # Input metadata (pre-split)
    INPUT_DIR = "./metadata"
    TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    VAL_PATH = os.path.join(INPUT_DIR, "validation.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

    # Working directory for this specific experiment (Idea 4)
    WORKING_DIR = "./working/idea_4"

    # Model checkpoints
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Cache files for preprocessed data (numpy format)
    TRAIN_CACHE_X = os.path.join(WORKING_DIR, "train_x.npy")
    TRAIN_CACHE_Y = os.path.join(WORKING_DIR, "train_y.npy")
    VAL_CACHE_X = os.path.join(WORKING_DIR, "val_x.npy")
    VAL_CACHE_Y = os.path.join(WORKING_DIR, "val_y.npy")
    TEST_CACHE_X = os.path.join(WORKING_DIR, "test_x.npy")

    # Scaler statistics cache
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler_stats.npz")

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Feature Engineering & Routing
    # ==========================================
    # Column names in the raw/processed dataframes
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"
    TIME_COL = "time_step"
    TARGET_COL = "pressure"

    # Feature Routing: Maps specific features to model branches
    # 1. TCN Branch (Resistive Stream):
    #    Models R * Flow. Needs fast dynamics: Control input and its derivatives.
    TCN_FEATURES = ["u_in", "u_in_diff1", "u_in_diff2"]

    # 2. LSTM Branch (Elastic Stream):
    #    Models Volume / C. Needs state accumulation: Control input and integral (volume).
    LSTM_FEATURES = ["u_in", "u_in_cumsum"]

    # 3. Skip Connection (Physics Injection):
    #    Static lung attributes and pre-computed physical interaction terms.
    #    Also includes u_out to help the final layer gate the expiratory phase.
    SKIP_FEATURES = ["R", "C", "R_flow", "C_volume", "u_out"]

    # Full list of derived features to generate during preprocessing
    # (Used by the Dataset class to know what to compute)
    DERIVED_FEATURES_LIST = [
        "u_in_diff1",  # First derivative (Finite Difference)
        "u_in_diff2",  # Second derivative (Acceleration)
        "u_in_cumsum",  # Integral (Volume proxy)
        "R_flow",  # Interaction: R * u_in
        "C_volume",  # Interaction: u_in_cumsum / C
    ]

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # TCN Branch
    TCN_CHANNELS = [32, 64, 64, 128]
    TCN_KERNEL_SIZE = 3
    TCN_DROPOUT = 0.1

    # LSTM Branch
    LSTM_HIDDEN_DIM = 128
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.1

    # Fusion
    FC_HIDDEN_DIM = 128

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 60
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    # Data Loading
    NUM_WORKERS = 4

    # Debugging
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Number of breaths to use if DEBUG is True

    @staticmethod
    def setup():
        """
        Creates necessary working and submission directories.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def get_input_dims():
        """
        Returns the input dimension for each model branch based on the feature lists.
        """
        return {
            "tcn": len(Config.TCN_FEATURES),
            "lstm": len(Config.LSTM_FEATURES),
            "skip": len(Config.SKIP_FEATURES),
        }
