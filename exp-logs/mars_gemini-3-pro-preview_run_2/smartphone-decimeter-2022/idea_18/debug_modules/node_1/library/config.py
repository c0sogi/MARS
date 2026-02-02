import os


class Config:
    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_18"
    SUBMISSION_DIR = "./submission"

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working and submission directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    RANDOM_SEED = 42

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Sliding window size (N)
    WINDOW_SIZE = 15

    # Scaling factors for simple metric conversion
    # 1 degree lat ~ 111320 meters
    LAT_SCALE_FACTOR = 111320.0
    # Longitude scaling depends on latitude, will be computed dynamically or approx
    # For simplicity in config, we might define a global approx or handle in code.
    # Here we just define the constant for Latitude.

    # -------------------------------------------------------------------------
    # Feature Definitions
    # -------------------------------------------------------------------------
    # Raw columns to load from GNSS files
    GNSS_COLS_TO_LOAD = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "Cn0DbHz",
        "RawPseudorangeUncertaintyMeters",
    ]

    # Kinematic Stream Features (Input to CNN)
    # These are derived features computed per timestep in the window
    # 1. Relative Coordinates (meters) centered on window midpoint
    # 2. Dynamics (Velocity/Deltas)
    # 3. Signal Metrics
    KINEMATIC_FEATURES = [
        "rel_lat_m",
        "rel_lon_m",
        "rel_alt_m",
        "vel_lat_m",
        "vel_lon_m",
        "vel_alt_m",
        "scaled_cn0",
        "scaled_uncertainty",
    ]

    # Sky Context Stream Features (Input to MLP)
    # These are aggregated statistics over the window
    SKY_FEATURES = [
        "mean_elevation",
        "std_elevation",
        "mean_azimuth",
        "std_azimuth",
        "mean_cn0",
        "std_cn0",
        "mean_uncertainty",
    ]

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Kinematic Stream (1D CNN)
    CNN_IN_CHANNELS = len(KINEMATIC_FEATURES)
    CNN_HIDDEN_CHANNELS = 64
    CNN_KERNEL_SIZE = 3
    CNN_LAYERS = 3
    CNN_DROPOUT = 0.1

    # Sky Context Stream (MLP)
    SKY_IN_DIM = len(SKY_FEATURES)
    SKY_HIDDEN_DIM = 32

    # Fusion Head (MLP)
    # Input dim will be (CNN_HIDDEN_CHANNELS * WINDOW_SIZE) + SKY_HIDDEN_DIM
    FUSION_HIDDEN_DIMS = [128, 64]
    OUTPUT_DIM = 2  # Delta East, Delta North (meters)

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 30  # Adjust based on convergence speed
    EARLY_STOPPING_PATIENCE = 5

    # Scheduler parameters
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 2
    SCHEDULER_MIN_LR = 1e-6

    # -------------------------------------------------------------------------
    # Caching Filenames
    # -------------------------------------------------------------------------
    CACHE_TRAIN_X_SEQ = os.path.join(WORK_DIR, "train_X_seq.npy")
    CACHE_TRAIN_X_SKY = os.path.join(WORK_DIR, "train_X_sky.npy")
    CACHE_TRAIN_Y = os.path.join(WORK_DIR, "train_y.npy")
    CACHE_TRAIN_META = os.path.join(WORK_DIR, "train_meta.parquet")

    CACHE_VAL_X_SEQ = os.path.join(WORK_DIR, "val_X_seq.npy")
    CACHE_VAL_X_SKY = os.path.join(WORK_DIR, "val_X_sky.npy")
    CACHE_VAL_Y = os.path.join(WORK_DIR, "val_y.npy")
    CACHE_VAL_META = os.path.join(WORK_DIR, "val_meta.parquet")

    CACHE_TEST_X_SEQ = os.path.join(WORK_DIR, "test_X_seq.npy")
    CACHE_TEST_X_SKY = os.path.join(WORK_DIR, "test_X_sky.npy")
    CACHE_TEST_META = os.path.join(WORK_DIR, "test_meta.parquet")

    SCALER_PATH = os.path.join(WORK_DIR, "scaler.json")
    MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")
