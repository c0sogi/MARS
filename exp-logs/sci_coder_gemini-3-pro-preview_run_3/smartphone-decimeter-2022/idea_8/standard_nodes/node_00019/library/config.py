import os


class Config:
    # -------------------------------------------------------------------------
    # Global Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    MAX_ROWS = 10000 if DEBUG else None  # Limit rows when debugging

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    OUTPUT_DIR = "./submission"

    # Ensure working and output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Paths (Base directories)
    TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DATA_DIR = os.path.join(INPUT_DIR, "test")

    # -------------------------------------------------------------------------
    # Caching Paths (Parquet files)
    # -------------------------------------------------------------------------
    # Features and Targets
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    TRAIN_TARGETS_PATH = os.path.join(WORKING_DIR, "train_targets.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    VAL_TARGETS_PATH = os.path.join(WORKING_DIR, "val_targets.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

    # Intermediate predictions (Raw ML output before smoothing)
    VAL_PREDS_RAW_PATH = os.path.join(WORKING_DIR, "val_preds_raw.parquet")
    TEST_PREDS_RAW_PATH = os.path.join(WORKING_DIR, "test_preds_raw.parquet")

    # Final Submission
    SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Feature Engineering Parameters
    # -------------------------------------------------------------------------
    # GNSS Signal Types to aggregate
    SIGNAL_TYPES = [
        "GPS_L1",
        "GPS_L5",
        "GAL_E1",
        "GAL_E5A",
        "GLO_G1",
        "BDS_B1I",
        "BDS_B1C",
        "BDS_B2A",
        "QZS_J1",
        "QZS_J5",
    ]

    # -------------------------------------------------------------------------
    # Model Hyperparameters (LightGBM)
    # -------------------------------------------------------------------------
    # Using L1 (MAE) objective to be robust against heavy-tailed outliers
    LGBM_PARAMS = {
        "objective": "regression_l1",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": -1,
        "random_state": SEED,
        "verbosity": -1,
    }

    N_ESTIMATORS = 5000
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    # -------------------------------------------------------------------------
    # Post-Processing Parameters (Kalman Filter & Smoothing)
    # -------------------------------------------------------------------------
    # Physics-based Innovation Gating
    # Threshold for innovation magnitude (meters).
    # If ||residual|| > threshold, the measurement is discarded (outlier).
    # This prevents multipath spikes from dragging the filter state.
    KF_GATE_THRESHOLD = 15.0

    # Process Noise (Q)
    # Variance of the acceleration noise (assuming Constant Velocity model)
    # Higher = system trusts dynamics less, follows measurements more
    KF_Q_SIGMA = 0.5

    # Measurement Noise (R)
    # Base variance for GNSS measurements
    # Higher = system trusts measurements less, smooths more
    KF_R_SIGMA = 5.0

    # Velocity Initialization (RANSAC)
    # Parameters for Robust Linear Regression to estimate initial velocity
    RANSAC_WINDOW_SECONDS = 5  # Use first N seconds of the track
    RANSAC_MIN_SAMPLES = 5  # Minimum points required to fit
    RANSAC_RESIDUAL_THRESHOLD = 2.0  # Inlier threshold in meters
