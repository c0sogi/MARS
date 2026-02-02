import os
import torch


class Config:
    # =========================================================================
    # Paths and Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching intermediate files (idea_4 specific)
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Processing Parameters
    # =========================================================================
    SEED = 42

    # WGS84 Ellipsoid Constants for Coordinate Conversions (Degrees <-> Meters)
    WGS84_A = 6378137.0
    WGS84_F = 1.0 / 298.257223563
    WGS84_B = WGS84_A * (1.0 - WGS84_F)

    # Columns to load from the raw device_gnss.csv files
    GNSS_COLS = [
        "utcTimeMillis",
        "Cn0DbHz",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "RawPseudorangeUncertaintyMeters",
        "Svid",
        "ConstellationType",
        "SignalType",
        "CodeType",
    ]

    # Aggregation Configuration for 1Hz alignment
    # Dictionary mapping raw column names to lists of aggregation functions
    GNSS_AGG_CONFIG = {
        "Cn0DbHz": ["mean", "max", "std"],
        "SvElevationDegrees": ["mean", "std"],
        "SvAzimuthDegrees": ["mean", "std"],
        "RawPseudorangeUncertaintyMeters": ["mean", "min", "max"],
        "Svid": ["count"],  # Proxy for satellite count
    }

    # Input dimension calculation based on the aggregations above:
    # 3 (Cn0) + 2 (El) + 2 (Az) + 3 (RawUnc) + 1 (Count) = 11 features
    IN_CHANNELS = 11

    # Target variables (Residuals in meters: North, East)
    TARGET_COLS = ["lat_res_m", "lon_res_m"]
    OUTPUT_CHANNELS = 2

    # =========================================================================
    # Model Hyperparameters (1D U-Net)
    # =========================================================================
    MODEL_NAME = "UNet1D"
    BASE_FILTERS = 32
    KERNEL_SIZE = 3
    DEPTH = 4  # Number of encoder/decoder blocks
    DROPOUT = 0.1

    # =========================================================================
    # Training Settings
    # =========================================================================
    EPOCHS = 50
    BATCH_SIZE = 16  # Batch size (number of drives/sequences)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # System settings
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100
