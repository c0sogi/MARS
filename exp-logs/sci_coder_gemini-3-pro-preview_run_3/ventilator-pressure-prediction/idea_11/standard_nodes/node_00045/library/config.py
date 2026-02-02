import os
import torch


class Config:
    """
    Global configuration for the Ventilator Pressure Prediction task.
    Implements settings for the Physically-Modulated Non-Causal Hybrid (PM-NC-Net).
    """

    # --------------------------------------------------------------------------
    # General Setup
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging

    # --------------------------------------------------------------------------
    # Directories
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"  # Specific cache directory for this idea
    SUBMISSION_DIR = "./submission"

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Metadata splits (guaranteed to be disjoint by breath_id)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "validation.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Raw sample submission
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoints
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    LAST_MODEL_PATH = os.path.join(WORKING_DIR, "last_model.pth")

    # Data Cache Files (Numpy format for speed and memory efficiency)
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_x.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_x.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_x.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Scaler Statistics
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler_stats.npz")

    # --------------------------------------------------------------------------
    # Feature Configuration
    # --------------------------------------------------------------------------
    # Features aligned with PM-NC-Net: PID, Lookahead, and Physics Interactions.

    # Dynamic Continuous Features (Subject to Robust Scaling)
    DYN_FEATURES = [
        "time_step",
        "u_in",
        "u_in_cumsum",  # Integral (PID)
        "u_in_diff1",  # Derivative (PID)
        "u_in_diff2",  # Acceleration (PID)
        "u_in_next",  # Lookahead t+1 (Non-Causal)
        "u_in_next2",  # Lookahead t+2 (Non-Causal)
        "area",  # Volume approximation (Integral of u_in * dt)
        "R_u_in",  # Physics Interaction: R * Flow
        "area_C",  # Physics Interaction: Volume / Compliance
    ]

    # Static Lung Attributes (Used for FiLM Modulation)
    STATIC_FEATURES = ["R", "C"]

    # Control/Mask Features
    CONTROL_FEATURES = ["u_out"]

    # Full Feature List (Order matters for tensor construction)
    FEATURE_COLS = DYN_FEATURES + STATIC_FEATURES + CONTROL_FEATURES

    # Target and Identifiers
    TARGET_COL = "pressure"
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"

    # --------------------------------------------------------------------------
    # Model Hyperparameters (PM-NC-Net)
    # --------------------------------------------------------------------------
    # Training Configuration
    BATCH_SIZE = 128  # Small batch size for gradient noise
    EPOCHS = 80  # Extended training budget
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0

    # Scheduler Configuration (ReduceLROnPlateau)
    SCHEDULER_PATIENCE = 7
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Architecture: Hybrid TCN + LSTM with FiLM
    # Branch 1: Modulated Non-Causal TCN (Resistive Stream)
    TCN_CHANNELS = 64
    TCN_KERNEL_SIZE = 7
    TCN_LAYERS = 4
    TCN_DROPOUT = 0.1

    # Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
    LSTM_HIDDEN_DIM = 512
    LSTM_LAYERS = 3
    LSTM_DROPOUT = 0.1

    # Fusion Head
    FC_HIDDEN_DIM = 256

    # --------------------------------------------------------------------------
    # Hardware & System
    # --------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def initialize(cls):
        """
        Initializes the working environment.
        Creates necessary directories for caching and submission.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
