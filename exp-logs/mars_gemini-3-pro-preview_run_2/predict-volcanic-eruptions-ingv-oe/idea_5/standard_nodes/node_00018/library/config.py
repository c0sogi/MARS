import os
import torch


class Config:
    """
    Configuration class for the Volcano Eruption Prediction project.
    Defines project-wide constants, hyperparameters, and file paths.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    PROJECT_NAME = "volcano_eruption_prediction"
    IDEA_NAME = "idea_5"
    SEED = 42

    # Compute
    # Adjust workers based on vCPU count (12 available)
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", IDEA_NAME)
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (Parquet/Numpy)
    # Used for caching engineered features and scalers
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Scaler Statistics
    STATS_SCALER_MEAN_PATH = os.path.join(WORKING_DIR, "stats_scaler_mean.npy")
    STATS_SCALER_SCALE_PATH = os.path.join(WORKING_DIR, "stats_scaler_scale.npy")

    # Target Normalization Statistics
    TARGET_MEAN_PATH = os.path.join(WORKING_DIR, "target_mean.npy")
    TARGET_STD_PATH = os.path.join(WORKING_DIR, "target_std.npy")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    NUM_SENSORS = 10
    SAMPLE_RATE = 100  # 60001 samples / 600 seconds = 100 Hz
    SEGMENT_LENGTH = 60001

    # Spectrogram Generation (Multi-Resolution)
    # View A: High Temporal Resolution (Short Window)
    # Captures transient events like sudden tremors
    N_FFT_SHORT = 256
    WIN_LENGTH_SHORT = 256

    # View B: High Frequency Resolution (Long Window)
    # Captures harmonic patterns and continuous rumbling
    N_FFT_LONG = 2048
    WIN_LENGTH_LONG = 2048

    # Shared Parameters
    # Hop length must be consistent to align time dimensions for stacking
    HOP_LENGTH = 256
    N_MELS = 128
    F_MIN = 0
    F_MAX = None  # Nyquist frequency
    TOP_DB = 80.0  # For AmplitudeToDB normalization

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "resnet34"
    PRETRAINED = True
    # Input Channels: 10 sensors * 2 views (Short/Long FFT) = 20 channels
    IN_CHANNELS = 20
    FC_DIM = 512  # Dimension of shared dense layer before head
    DROPOUT = 0.5

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 20

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    DEBUG = False  # Set to True to use a subset of data for quick testing
    DEBUG_SAMPLE_SIZE = 100
