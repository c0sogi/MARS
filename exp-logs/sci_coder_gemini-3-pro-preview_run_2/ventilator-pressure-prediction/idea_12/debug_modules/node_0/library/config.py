import os
import torch


class Config:
    # ==========================================
    # Global Settings & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run in debug mode (fewer epochs, smaller dataset)

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs

    # ==========================================
    # Data Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Raw Data
    TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
    TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache / Processed Data
    # Using Parquet for efficient storage of processed dataframes
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "train_processed.parquet")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "val_processed.parquet")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "test_processed.parquet")

    # Artifacts
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.npy")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Feature Engineering
    # ==========================================
    # List of features to be used by the model.
    # This ensures consistency between dataset generation and model input.
    # 1. Raw: time_step, u_in, u_out, R, C
    # 2. Physics: u_in_cumsum (Volume), R_u_in (Resistive), vol_C (Elastic)
    # 3. Dynamics: u_in_lag1, u_in_lag2, u_in_diff1, u_in_diff2
    # 4. u_out is categorical/binary, others are continuous.

    FEATURE_COLS = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_cumsum",
        "u_in_lag1",
        "u_in_lag2",
        "u_in_diff1",
        "u_in_diff2",
        "R_u_in",
        "vol_C",
    ]

    INPUT_DIM = len(FEATURE_COLS)
    SEQ_LEN = 80  # Standard breath length in this dataset

    # ==========================================
    # Model Architecture (DP-GI-BiLSTM)
    # ==========================================
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 4
    BIDIRECTIONAL = True

    # Dual-Path Injection Settings
    # Path A is Identity (Raw Features)
    # Path B is GLU Projection
    GLU_PROJECTION_DIM = 128

    # Regularization
    DROPOUT = 0.1  # Applied within recurrent blocks, NOT on injection path

    # ==========================================
    # Training Protocol (Stretched-Horizon)
    # ==========================================
    EPOCHS = 200
    BATCH_SIZE = 512  # A100 40GB can handle large batches

    # Optimizer & Scheduler
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    T_MAX = 200  # Matches EPOCHS for stretched cosine annealing
    ETA_MIN = 1e-5

    # Loss Weights
    # Focus on inspiratory phase (u_out=0), reduced weight for expiratory
    LOSS_INSPIRATORY_WEIGHT = 1.0
    LOSS_EXPIRATORY_WEIGHT = 0.1

    # ==========================================
    # Debugging / Development Overrides
    # ==========================================
    DEBUG_EPOCHS = 2
    DEBUG_SAMPLE_SIZE = 1000  # Number of breaths to sample in debug mode

    @classmethod
    def get_epochs(cls):
        """Returns number of epochs based on debug flag."""
        return cls.DEBUG_EPOCHS if cls.DEBUG else cls.EPOCHS

    @classmethod
    def get_sample_size(cls):
        """Returns sample size (None for full data) based on debug flag."""
        return cls.DEBUG_SAMPLE_SIZE if cls.DEBUG else None
