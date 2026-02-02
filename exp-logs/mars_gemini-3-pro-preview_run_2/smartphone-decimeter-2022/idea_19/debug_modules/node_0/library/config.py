import os


class Config:
    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging/testing

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_19"
    SUBMISSION_DIR = "./submission"

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    # Window size N (odd number recommended to have a clear center)
    # e.g., 15 means center epoch +/- 7 epochs context
    WINDOW_SIZE = 15

    # Columns to load from GNSS files
    # We load WLS positions to compute relative coordinates and signal info for context
    GNSS_COLS = [
        "utcTimeMillis",
        "WlsPositionXEcefMeters",
        "WlsPositionYEcefMeters",
        "WlsPositionZEcefMeters",
        "SvElevationDegrees",
        "SvAzimuthDegrees",
        "Cn0DbHz",
        "RawPseudorangeUncertaintyMeters",
    ]

    # Columns to load from IMU files
    IMU_COLS = [
        "utcTimeMillis",
        "MessageType",
        "MeasurementX",
        "MeasurementY",
        "MeasurementZ",
    ]

    # --------------------------------------------------------------------------
    # Feature Definitions
    # --------------------------------------------------------------------------
    # Features per timestep in the trajectory block (Flattened later)
    # These correspond to the processed feature names generated in the pipeline
    TRAJ_FEATURES = [
        "rel_lat_m",
        "rel_lon_m",
        "rel_alt_m",  # Position relative to window center (Meters)
        "vel_lat_m",
        "vel_lon_m",
        "vel_alt_m",  # Velocity (First difference)
        "cn0",
        "unc_m",  # Signal Quality
    ]

    # Environmental Context Features (Aggregated over window)
    ENV_FEATURES = ["mean_elev", "std_elev", "mean_azim", "std_azim"]

    # Inertial Context Features (Aggregated over window)
    IMU_FEATURES = ["mean_acc_mag", "std_acc_mag", "mean_gyro_mag", "std_gyro_mag"]

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10

    # MLP Architecture
    HIDDEN_DIM = 512
    NUM_LAYERS = 4
    DROPOUT = 0.1

    def __init__(self):
        # Ensure working directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)


# Instantiate config to be imported by other modules
config = Config()
