import os
import torch


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    # Input data (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output/Working data (Writeable)
    # Using specific subdirectory for this idea to avoid conflicts
    WORKING_DIR = "./working/idea_10"
    SUBMISSION_DIR = "./submission"

    # Create writeable directories immediately
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths (Preferred over raw input for splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Final Submission Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Cache Configuration
    # =========================================================================
    # Flag to trigger automated cache invalidation (Lesson 6)
    FORCE_REGENERATE_CACHE = True

    # Cache File Paths
    # We use .npy for efficient storage of processed numpy arrays
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_x.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_x.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_x.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # Scaler Statistics Cache (Median/IQR for RobustScaler)
    CACHE_SCALER = os.path.join(WORKING_DIR, "scaler_stats.npz")

    # =========================================================================
    # Feature Configuration
    # =========================================================================
    # Unified configuration for generation and model input (Lesson 5)
    # Includes Dynamic PID features and Physics Interactions
    FEATURE_COLS = [
        "time_step",  # Time
        "u_in",  # Control Input (Proportional)
        "u_out",  # Control Input (Expiratory Phase)
        "R",  # Lung Resistance
        "C",  # Lung Compliance
        "u_in_cumsum",  # Integral (Proxy for Volume)
        "u_in_diff1",  # Derivative (Proxy for Flow)
        "u_in_diff2",  # Acceleration
        "R_u_in",  # Interaction: Resistive Pressure (R * Flow_proxy)
        "vol_C",  # Interaction: Elastic Pressure (Volume / C)
    ]

    # The target variable to predict
    TARGET_COL = "pressure"

    # Identification columns
    ID_COL = "id"
    BREATH_ID_COL = "breath_id"

    # =========================================================================
    # Model Hyperparameters (NCP-Net)
    # =========================================================================
    # General
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Training
    # Small batch size (128) for gradient noise injection (Lesson 18)
    BATCH_SIZE = 128
    # Extended training budget for hybrid convergence (Lesson 39)
    EPOCHS = 80

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    CLIP_GRAD_NORM = 1.0

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # Architecture: Non-Causal Pyramidal TCN (Branch 1)
    # Wide kernel (7) for smoothing physical signals (Lesson 26)
    TCN_KERNEL_SIZE = 7
    # Base channels, will be doubled in pyramidal structure (Lesson 24)
    TCN_BASE_CHANNELS = 64
    # Number of layers (determines dilation 1, 2, 4, 8...)
    TCN_LAYERS = 4
    TCN_DROPOUT = 0.1

    # Architecture: Bidirectional LSTM (Branch 2)
    # High capacity for integral dynamics (Lesson 32)
    LSTM_HIDDEN_DIM = 512
    LSTM_LAYERS = 3
    LSTM_DROPOUT = 0.1

    # Fusion Head
    # Deep dense layers after concatenation
    HEAD_HIDDEN_DIMS = [512, 256]

    # =========================================================================
    # Data Processing Constants
    # =========================================================================
    # Sequence length (breaths are approx 80 steps, padded/truncated if needed)
    # In this dataset, breaths are typically 80 steps.
    MAX_SEQ_LEN = 80

    # Debugging
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLES = 1000
