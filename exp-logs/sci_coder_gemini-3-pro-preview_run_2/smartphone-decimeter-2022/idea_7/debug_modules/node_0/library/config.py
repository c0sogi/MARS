import os


class Config:
    """
    Centralized configuration for the Local-Attention Transformer (LAT) solution.
    Handles file paths, hyperparameters, and data settings.
    """

    # ---------------------------------------------------------
    # File Paths & Directories
    # ---------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (for deterministic data processing)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.parquet")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler.json")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "lat_model.pth")

    # Final Submission
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ---------------------------------------------------------
    # Data Preprocessing Parameters
    # ---------------------------------------------------------
    # Sliding window size (number of epochs). Must be odd to define a center.
    WINDOW_SIZE = 11

    # Approximate conversion factor for degrees to meters (Latitude)
    METERS_PER_DEG_LAT = 111320.0

    # Raw GNSS columns to load from device_gnss.csv
    RAW_GNSS_COLS = [
        "tripId",
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "Cn0DbHz",
        "RawPseudorangeUncertaintyMeters",
        "Svid",
    ]

    # Derived Feature Columns used as input to the model
    # 1. rel_lat_m: Latitude relative to window center, scaled to meters
    # 2. rel_lon_m: Longitude relative to window center, scaled to meters
    # 3. vel_lat_m: First-order difference of Latitude, scaled to meters
    # 4. vel_lon_m: First-order difference of Longitude, scaled to meters
    # 5. vel_alt_m: First-order difference of Altitude (meters)
    # 6. mean_cn0: Mean Carrier-to-Noise density
    # 7. mean_unc: Mean Raw Pseudorange Uncertainty
    # 8. sat_count: Number of satellites
    INPUT_FEATURES = [
        "rel_lat_m",
        "rel_lon_m",
        "vel_lat_m",
        "vel_lon_m",
        "vel_alt_m",
        "mean_cn0",
        "mean_unc",
        "sat_count",
    ]

    # Input dimension for the model embedding layer
    INPUT_DIM = len(INPUT_FEATURES)

    # Output dimension: Predicted residual corrections (Delta East, Delta North) in meters
    OUTPUT_DIM = 2

    # ---------------------------------------------------------
    # Model Hyperparameters (Local-Attention Transformer)
    # ---------------------------------------------------------
    D_MODEL = 128  # Embedding dimension
    NHEAD = 4  # Number of attention heads
    NUM_ENCODER_LAYERS = 3  # Number of transformer encoder layers
    DIM_FEEDFORWARD = 256  # Dimension of the feedforward network
    DROPOUT = 0.1  # Dropout rate

    # ---------------------------------------------------------
    # Training Parameters
    # ---------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 30
    PATIENCE = 5  # Early stopping patience

    # ---------------------------------------------------------
    # Debugging / Development
    # ---------------------------------------------------------
    DEBUG = False  # Set to True to use a smaller subset of data
    DEBUG_SAMPLE_SIZE = 50  # Number of trips to use in debug mode
