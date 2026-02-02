import os


class Config:
    """
    Configuration class for the Distribution-Aware 1D ResUNet pipeline.
    """

    # --- Directory Paths ---
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_5")
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- Reproducibility ---
    SEED = 42

    # --- Data Preprocessing ---
    # Features to load from raw GNSS logs
    GNSS_COLS = [
        "utcTimeMillis",
        "Cn0DbHz",
        "SvElevationDegrees",
        "RawPseudorangeUncertaintyMeters",
    ]

    # Aggregation specifications for creating the input sequence
    # Keys are column names, Values are lists of statistics
    # Note: 'count' is implicitly calculated to represent Satellite Count
    AGGREGATION_SPECS = {
        "Cn0DbHz": ["mean", "std", "min", "max", "median", "q25", "q75"],
        "SvElevationDegrees": ["mean", "min", "max"],
        "RawPseudorangeUncertaintyMeters": ["mean"],
    }

    # WGS84 Ellipsoid Constants for coordinate conversion (Lat/Lon <-> East/North)
    WGS84_A = 6378137.0
    WGS84_B = 6356752.314245
    WGS84_F = 1 / 298.257223563

    # --- Model Hyperparameters (1D ResUNet) ---
    # Input Channels Calculation:
    # Cn0DbHz (7 stats) + SvElevationDegrees (3 stats) +
    # RawPseudorangeUncertaintyMeters (1 stat) + SatCount (1 stat) = 12
    IN_CHANNELS = 12

    # Output Channels: 2 (Delta East in meters, Delta North in meters)
    OUT_CHANNELS = 2

    # Architecture dimensions
    MODEL_CHANNELS = 64
    MODEL_DEPTH = 4
    KERNEL_SIZE = 3
    DROPOUT = 0.1

    # --- Training Parameters ---
    BATCH_SIZE = (
        8  # Small batch size due to variable and potentially long sequence lengths
    )
    LEARNING_RATE = 1e-3  # Initial learning rate for AdamW
    WEIGHT_DECAY = 1e-2  # Weight decay for regularization
    EPOCHS = 50  # Maximum number of epochs
    EARLY_STOPPING_PATIENCE = (
        10  # Stop if validation loss doesn't improve for 10 epochs
    )
    NUM_WORKERS = 4  # Number of dataloader workers

    # --- Debugging ---
    DEBUG = False  # Set to True to run on a small subset of data for testing
