import os
import torch


class Config:
    """
    Global configuration for the Spectrally-Enhanced SE-ResNet Hybrid model.
    Handles paths, hyperparameters, and constants for data processing and training.
    """

    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths for Deterministic Data Processing
    # Using parquet for tabular features and npy for scaler statistics
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    STATS_SCALER_MEAN_PATH = os.path.join(WORKING_DIR, "stats_scaler_mean.npy")
    STATS_SCALER_SCALE_PATH = os.path.join(WORKING_DIR, "stats_scaler_scale.npy")
    TARGET_MEAN_PATH = os.path.join(WORKING_DIR, "target_mean.npy")
    TARGET_STD_PATH = os.path.join(WORKING_DIR, "target_std.npy")

    # --------------------------------------------------------------------------
    # Signal Processing & Spectrogram Parameters
    # --------------------------------------------------------------------------
    SAMPLING_RATE = 100  # Inferred: 60001 samples / 10 minutes
    N_MELS = 128  # Frequency resolution for Mel Spectrogram
    N_FFT = 1024  # FFT window size
    HOP_LENGTH = 256  # Stride for STFT
    TOP_DB = 80  # Dynamic range for AmplitudeToDB

    # --------------------------------------------------------------------------
    # Model Architecture Parameters
    # --------------------------------------------------------------------------
    NUM_SENSORS = 10  # Number of seismic sensors
    MLP_HIDDEN_DIM = 256  # Hidden dimension for the tabular MLP branch

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32  # Optimized for A100 GPU
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW optimizer
    NUM_EPOCHS = 100  # Maximum training epochs
    PATIENCE = 20  # Early stopping patience

    # Scheduler Parameters (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-6

    # --------------------------------------------------------------------------
    # Runtime & Debugging Controls
    # --------------------------------------------------------------------------
    DEBUG = False  # Set True to run on a small subset for testing
    DEBUG_SIZE = 100  # Number of samples to use when DEBUG=True
    NUM_WORKERS = 4  # Number of DataLoader workers (12 vCPUs available)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
