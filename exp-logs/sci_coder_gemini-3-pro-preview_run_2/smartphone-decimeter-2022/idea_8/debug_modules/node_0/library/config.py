import os
import torch


class Config:
    """
    Configuration class for the Short-Window Relative-State Bi-GRU model.
    """

    # -------------------------------------------------------------------------
    # Directory and File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (numpy arrays and json for scaler)
    CACHE_TRAIN_X = os.path.join(WORKING_DIR, "train_X.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_VAL_X = os.path.join(WORKING_DIR, "val_X.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_TEST_X = os.path.join(WORKING_DIR, "test_X.npy")
    CACHE_TEST_META = os.path.join(
        WORKING_DIR, "test_meta.parquet"
    )  # Caching test metadata for reconstruction
    CACHE_SCALER = os.path.join(WORKING_DIR, "scaler.json")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Sliding window size (N epochs)
    WINDOW_SIZE = 15

    # Approximate meters per degree for simple scaling
    LAT_SCALE = 111320.0

    # Features to be used in the model input
    # 1. rel_lat_m: Latitude relative to window center (meters)
    # 2. rel_lon_m: Longitude relative to window center (meters)
    # 3. vel_lat_m: Latitude velocity (m/s)
    # 4. vel_lon_m: Longitude velocity (m/s)
    # 5. vel_alt_m: Altitude velocity (m/s)
    # 6. raw_pr_unc: Mean Raw Pseudorange Uncertainty
    # 7. cn0: Mean Cn0DbHz
    # 8. sat_count: Number of satellites
    FEATURE_COLUMNS = [
        "rel_lat_m",
        "rel_lon_m",
        "vel_lat_m",
        "vel_lon_m",
        "vel_alt_m",
        "raw_pr_unc",
        "cn0",
        "sat_count",
    ]

    INPUT_DIM = len(FEATURE_COLUMNS)
    OUTPUT_DIM = 2  # Predicted residual: dLat_m, dLon_m

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    BIDIRECTIONAL = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 30  # Sufficient for convergence with early stopping
    PATIENCE = 6  # Early stopping patience

    # Hardware settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
