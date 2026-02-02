import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_META = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet format preferred over pickle/npy for dataframes)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_processed.parquet")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_processed.parquet")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_processed.parquet")
    SCALER_CACHE = os.path.join(WORKING_DIR, "scaler.joblib")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data & Feature Engineering
    # ==========================================
    # Random Seed
    SEED = 42

    # Data Loading
    NUM_WORKERS = 8
    USE_CACHE = True  # Set to False to force re-processing
    DEBUG = False  # Set to True to use a small subset of data

    # Feature Definitions
    # Continuous features to be scaled
    CONT_FEATURES = ["time_step", "u_in", "R", "C"]
    # Binary/Categorical features
    CAT_FEATURES = ["u_out"]

    # Engineering Parameters
    LAG_STEPS = [1, 2, 3]  # Lags for u_in
    DIFF_STEPS = [1, 2]  # Finite differences for u_in

    # List of all feature names that will be generated and used by the model
    # This list must match the output of the preprocessing pipeline
    SELECTED_FEATURES = [
        "time_step",
        "u_in",
        "u_out",
        "R",
        "C",
        "u_in_cumsum",  # Integrated volume
        "R_u_in",  # Interaction: Resistance * Flow
        "u_in_cumsum_C",  # Interaction: Volume / Compliance
        "u_in_lag1",
        "u_in_lag2",
        "u_in_lag3",
        "u_in_diff1",
        "u_in_diff2",
        "u_in_diff3",
        "u_in_diff4",  # Acceleration/Jerk proxies
    ]

    # Input dimension calculation (dynamically updated if needed, but good to have a base)
    INPUT_DIM = len(SELECTED_FEATURES)

    # ==========================================
    # Model Architecture (SC-GI-BiLSTM)
    # ==========================================
    LSTM_HIDDEN_DIM = 512
    LSTM_LAYERS = 4
    PROJECTION_DIM = 512
    DROPOUT = 0.0  # Explicitly 0.0 for the context path
    BIDIRECTIONAL = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 200  # Extended duration for long-tail convergence
    BATCH_SIZE = 512  # Large batch size for A100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1000.0

    # Scheduler
    T_MAX = EPOCHS  # For Cosine Annealing
    ETA_MIN = 1e-5

    # Early Stopping
    PATIENCE = 20

    # Loss Weights
    W_INSPIRATORY = 1.0
    W_EXPIRATORY = 0.1

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
