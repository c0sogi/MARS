import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for Ventilator Pressure Prediction (Idea 6).
    Strategy: Multi-Scale CNN-LSTM with Channel-Wise Attention (SE-Residuals).
    """

    # --------------------------------------------------------------------------
    # Experiment Control
    # --------------------------------------------------------------------------
    EXP_ID = "idea_6"
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging

    # --------------------------------------------------------------------------
    # Directories & File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXP_ID)

    # Ensure working directory exists for caching and artifacts
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data Paths (Mapped to Metadata Splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Artifact Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Scaler Cache Paths
    SCALER_CENTER_PATH = os.path.join(WORKING_DIR, "scaler_center.npy")
    SCALER_SCALE_PATH = os.path.join(WORKING_DIR, "scaler_scale.npy")

    # --------------------------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------------------------
    SEQ_LEN = 80

    # Rich Feature Set defined in Strategy
    # Includes: Raw signals, Physics attributes, Lags, Diffs, and Interactions
    FEATURES = [
        "u_in",
        "u_out",
        "time_step",
        "R",
        "C",  # Raw & Attributes
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_lag4",  # Dynamics
        "u_in_diff1",
        "u_in_diff2",  # Acceleration/Inertia
        "area",  # Cumulative Volume (Integral of u_in)
        "u_in_R",  # Interaction: Flow Pressure (u_in * R)
        "area_div_C",  # Interaction: Elastic Pressure (Volume / C)
    ]

    INPUT_DIM = len(FEATURES)

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    # Multi-Scale Stem (Inception-1D)
    CNN_FILTERS = 64
    CNN_KERNELS = [3, 5, 7]

    # Backbone (Deep SE-Residual Bi-LSTM)
    LSTM_HIDDEN = 512
    LSTM_LAYERS = 4
    BIDIRECTIONAL = True

    # Squeeze-and-Excitation
    SE_RATIO = 16

    # Regularization
    DROPOUT = 0.1

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    EPOCHS = 20
    BATCH_SIZE = 512

    # Optimization (AdamW + OneCycleLR)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    CLIP_GRAD = 1.0

    # Scheduler Settings
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # --------------------------------------------------------------------------
    # Hardware & Reproducibility
    # --------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
