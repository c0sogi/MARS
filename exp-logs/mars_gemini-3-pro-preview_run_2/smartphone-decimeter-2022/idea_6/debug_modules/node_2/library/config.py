import os
import torch


class Config:
    """
    Configuration class for the Relative-Trajectory 1D-CNN pipeline.
    Defines constants, paths, and hyperparameters.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache File Paths (Data)
    # We use .npy for dense tensor data and .parquet for metadata frames
    TRAIN_X_CACHE = os.path.join(WORKING_DIR, "train_X.npy")
    TRAIN_Y_CACHE = os.path.join(WORKING_DIR, "train_y.npy")
    TRAIN_META_CACHE = os.path.join(WORKING_DIR, "train_meta.parquet")

    VAL_X_CACHE = os.path.join(WORKING_DIR, "val_X.npy")
    VAL_Y_CACHE = os.path.join(WORKING_DIR, "val_y.npy")
    VAL_META_CACHE = os.path.join(WORKING_DIR, "val_meta.parquet")

    TEST_X_CACHE = os.path.join(WORKING_DIR, "test_X.npy")
    TEST_META_CACHE = os.path.join(WORKING_DIR, "test_meta.parquet")

    # Cache Files (Artifacts)
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler_stats.json")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Window size for the 1D CNN (must be odd to have a distinct center)
    # 11 epochs corresponds to roughly +/- 5 seconds of context
    WINDOW_SIZE = 11

    # Debugging: Set to an integer (e.g., 1000) to limit dataset size for testing, or None for full data
    DEBUG_SAMPLE_SIZE = None

    # Input Features (Channels)
    # The model expects 9 channels per time step in the window:
    # 1. Relative East (m) - Position relative to window center baseline
    # 2. Relative North (m)
    # 3. Relative Up (m)
    # 4. Velocity East (m/s) - First order difference
    # 5. Velocity North (m/s)
    # 6. Velocity Up (m/s)
    # 7. Mean Cn0 (Signal Strength)
    # 8. Mean Uncertainty (Raw Pseudorange Uncertainty)
    # 9. Satellite Count
    NUM_INPUT_FEATURES = 9

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    CNN_HIDDEN_CHANNELS = [64, 128, 256]  # Filters for each conv block
    CNN_KERNEL_SIZE = 3
    FC_HIDDEN_DIM = 128
    DROPOUT_RATE = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 5

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Physics Constants
    # -------------------------------------------------------------------------
    # WGS84 Ellipsoid constants for coordinate conversion
    WGS84_A = 6378137.0
    WGS84_E2 = 6.69437999014e-3
