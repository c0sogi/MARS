import os


class Config:
    """
    Global configuration for the Doppler-Aided Residual Boosting pipeline.
    """

    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for debugging

    # -------------------------------------------------------------------------
    # Directory Structure & File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Create necessary output directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Caching Paths (Parquet files for processed features)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # WGS84 Ellipsoid & Physical Constants
    # -------------------------------------------------------------------------
    # Semi-major axis (meters)
    WGS84_A = 6378137.0
    # Flattening
    WGS84_F = 1.0 / 298.257223563
    # Semi-minor axis (derived)
    WGS84_B = WGS84_A * (1.0 - WGS84_F)
    # First eccentricity squared (derived)
    WGS84_E2 = WGS84_F * (2 - WGS84_F)

    # Earth rotation rate (rad/s)
    OMEGA_E = 7.2921151467e-5
    # Speed of light in vacuum (m/s)
    LIGHT_SPEED = 299792458.0

    # -------------------------------------------------------------------------
    # LightGBM Hyperparameters
    # -------------------------------------------------------------------------
    LGBM_PARAMS = {
        "objective": "mae",  # Mean Absolute Error for robustness to outliers
        "boosting_type": "gbdt",
        "n_estimators": 5000,  # Maximum number of trees
        "learning_rate": 0.05,  # Learning rate
        "num_leaves": 128,  # Max leaves per tree
        "max_depth": -1,  # No limit on depth
        "colsample_bytree": 0.8,  # Feature subsampling
        "subsample": 0.8,  # Row subsampling
        "subsample_freq": 1,
        "reg_alpha": 0.5,  # L1 regularization
        "reg_lambda": 0.5,  # L2 regularization
        "random_state": SEED,
        "n_jobs": -1,  # Use all available cores
        "verbose": -1,  # Silent mode
    }

    # Training Loop Parameters
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    # -------------------------------------------------------------------------
    # Kalman Filter / Smoothing Parameters
    # -------------------------------------------------------------------------
    # Gate threshold in meters for the Innovation step.
    # Measurements (ML predictions) deviating more than this from the
    # Doppler-predicted state will be rejected to handle multipath outliers.
    KF_GATE_THRESHOLD = 15.0

    # -------------------------------------------------------------------------
    # Feature Engineering Parameters
    # -------------------------------------------------------------------------
    # Minimum Carrier-to-Noise density (dB-Hz) to consider a signal valid
    # for Doppler velocity estimation.
    DOPPLER_CN0_THRESH = 20.0
