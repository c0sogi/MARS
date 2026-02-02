import os
import torch


class Config:
    """
    Global configuration for the Lookahead-Augmented Non-Causal Hybrid (LAN-Net) pipeline.
    """

    # --------------------------------------------------------------------------
    # 1. Directory & File Setup
    # --------------------------------------------------------------------------
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"

    # Ensure the working directory exists for caching and artifacts
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Data Paths (Using metadata splits for consistency)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "validation.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Artifact Paths
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pth")
    LAST_MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "last_model.pth")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler_stats.npz")
    SUBMISSION_FILE = os.path.join(WORKING_DIR, "submission.csv")

    # Cache Paths (for numpy arrays)
    TRAIN_CACHE_X = os.path.join(WORKING_DIR, "train_x.npy")
    TRAIN_CACHE_Y = os.path.join(WORKING_DIR, "train_y.npy")
    TRAIN_CACHE_U_OUT = os.path.join(WORKING_DIR, "train_u_out.npy")

    VAL_CACHE_X = os.path.join(WORKING_DIR, "val_x.npy")
    VAL_CACHE_Y = os.path.join(WORKING_DIR, "val_y.npy")
    VAL_CACHE_U_OUT = os.path.join(WORKING_DIR, "val_u_out.npy")

    TEST_CACHE_X = os.path.join(WORKING_DIR, "test_x.npy")
    TEST_CACHE_IDS = os.path.join(WORKING_DIR, "test_ids.npy")
    TEST_CACHE_U_OUT = os.path.join(WORKING_DIR, "test_u_out.npy")

    # --------------------------------------------------------------------------
    # 2. Data Configuration
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing

    # Column Definitions
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"
    TIME_COL = "time_step"
    TARGET_COL = "pressure"

    # Feature Engineering Configuration
    # The LAN-Net architecture requires specific inputs:
    # - PID State: u_in, u_in_cumsum (Volume), u_in_diff (Acceleration)
    # - Static Physics: R, C
    # - Interactions: R_u_in (R * u_in), vol_C (Volume / C)
    # - Lookahead: u_in_next1 (t+1), u_in_next2 (t+2), u_in_diff_next1 (diff t+1)
    # - Control: u_out (Used for masking and phase awareness)

    FEATURE_COLS = [
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_cumsum",  # Integral / Volume
        "u_in_diff",  # Derivative / Acceleration
        "R_u_in",  # Interaction: Resistive Pressure proxy
        "vol_C",  # Interaction: Elastic Pressure proxy
        "u_in_next1",  # Lookahead t+1
        "u_in_next2",  # Lookahead t+2
        "u_in_diff_next1",  # Lookahead derivative t+1
    ]

    INPUT_DIM = len(FEATURE_COLS)

    # --------------------------------------------------------------------------
    # 3. Model Hyperparameters
    # --------------------------------------------------------------------------
    # Training
    BATCH_SIZE = 128  # Small batch size for gradient noise regularization
    EPOCHS = 80  # Extended budget for hybrid convergence
    LEARNING_RATE = 1e-3  # Base learning rate
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 4

    # Architecture: LAN-Net (Lookahead-Augmented Non-Causal Hybrid)

    # Branch 1: Non-Causal Pyramidal TCN (Resistive Stream)
    TCN_KERNEL_SIZE = 7  # Wide kernel for smoothing
    TCN_CHANNELS = [64, 128, 256, 512]  # Pyramidal scaling
    TCN_DROPOUT = 0.1

    # Branch 2: High-Capacity Bidirectional LSTM (Elastic Stream)
    LSTM_HIDDEN_SIZE = 512
    LSTM_LAYERS = 3
    LSTM_BIDIRECTIONAL = True

    # --------------------------------------------------------------------------
    # 4. Runtime
    # --------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
