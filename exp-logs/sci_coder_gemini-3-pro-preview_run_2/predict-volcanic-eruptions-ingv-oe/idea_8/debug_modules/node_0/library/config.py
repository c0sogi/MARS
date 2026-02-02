import os
import torch


class Config:
    """
    Configuration class for the Volcano Eruption Prediction task.
    Implements parameters for Idea 8: Stabilized SE-ResNet34 Hybrid with Dual-Domain Feature Injection.
    """

    # --------------------------------------------------------------------------
    # General Configuration
    # --------------------------------------------------------------------------
    PROJECT_NAME = "volcano_eruption_prediction"
    IDEA_NAME = "idea_8"
    SEED = 42

    # Debugging flags to control dataset size and runtime
    # Set DEBUG = True to run on a small subset for testing pipeline
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use when DEBUG is True

    # --------------------------------------------------------------------------
    # Directory Paths
    # --------------------------------------------------------------------------
    # Input Data (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Read/Write)
    # Stores cache, models, and intermediate outputs for this specific idea
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # File Paths for Artifacts
    # --------------------------------------------------------------------------
    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "model.pth")

    # Cached Features (Parquet format for tabular/metadata)
    # These store the extracted dual-domain features (Time/Freq stats)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Target Scaling Statistics (Numpy)
    TARGET_MEAN_PATH = os.path.join(WORKING_DIR, "target_mean.npy")
    TARGET_STD_PATH = os.path.join(WORKING_DIR, "target_std.npy")

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    NUM_SENSORS = 10
    SAMPLE_RATE = 100  # 100 Hz (derived from 60001 samples / 600 seconds)
    SIGNAL_LENGTH = 60001  # Exact number of rows in CSV files

    # Spectrogram Parameters (Log-Mel)
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 256  # Results in approx 235 time steps
    TOP_DB = 80
    NORM_TYPE = "slaney"  # Critical fix: Slaney area normalization

    # --------------------------------------------------------------------------
    # Model Architecture Parameters
    # --------------------------------------------------------------------------
    # CNN Branch
    BACKBONE = "resnet34"
    PRETRAINED = True
    USE_SE = True  # Enable Squeeze-and-Excitation blocks
    IN_CHANNELS = 10  # One channel per sensor

    # MLP Branch (Dual-Domain Features)
    MLP_HIDDEN_LAYERS = [256, 128]
    MLP_DROPOUT = 0.2

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 50  # Maximum epochs (Early stopping will likely trigger sooner)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # For AdamW

    # Optimization Strategy
    WARMUP_EPOCHS = 5  # Linear warmup duration
    PATIENCE = 20  # Early stopping patience
    SCHEDULER_FACTOR = 0.1  # ReduceLROnPlateau factor
    SCHEDULER_PATIENCE = 5  # ReduceLROnPlateau patience

    # --------------------------------------------------------------------------
    # Hardware & Runtime
    # --------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
