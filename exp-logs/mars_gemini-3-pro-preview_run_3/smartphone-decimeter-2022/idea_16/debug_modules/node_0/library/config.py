import os
import numpy as np


class Config:
    """
    Global configuration for the GNSS Positioning Task.
    Implements settings for 'Geometry-Projected Boosting with Carrier-Phase Trajectory Alignment'.
    """

    # -------------------------------------------------------------------------
    # 1. Directories & File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")
    METADATA_DIR = "./metadata"

    # Generated Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory for Caching Intermediate Data (Parquet/Numpy)
    WORKING_DIR = "./working/idea_16"
    CACHE_DIR = WORKING_DIR

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure mutable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Global Constants & Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # Debugging / Development Mode
    # Set DEBUG = True to run on a small subset of data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000  # Number of rows to sample if DEBUG is True

    # -------------------------------------------------------------------------
    # 3. GNSS Physics & Geodesy
    # -------------------------------------------------------------------------
    # WGS84 Ellipsoid Parameters
    WGS84_A = 6378137.0  # Semi-major axis (meters)
    WGS84_F = 1.0 / 298.257223563  # Flattening
    WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis

    # Physical Constants
    LIGHT_SPEED = 299792458.0  # m/s

    # GNSS Frequencies (Hz)
    GPS_L1_HZ = 1575.42e6
    GPS_L5_HZ = 1176.45e6
    GLO_G1_HZ = 1602.0e6
    GAL_E1_HZ = 1575.42e6
    GAL_E5A_HZ = 1176.45e6
    BDS_B1I_HZ = 1561.098e6

    # Carrier Phase State Bitmasks (Android GnssMeasurement)
    # Used to filter valid accumulated delta range (ADR) for TDCP
    ADR_STATE_VALID = 1 << 0
    ADR_STATE_RESET = 1 << 1
    ADR_STATE_CYCLE_SLIP = 1 << 2
    ADR_STATE_HALF_CYCLE_RESOLVED = 1 << 3

    # Criteria for valid TDCP measurement:
    # Must be VALID, not RESET, not CYCLE_SLIP.
    # Half cycle resolved is preferred but not strictly strictly required for delta-phase if wavelength is handled.
    # Here we enforce strict validity.
    TDCP_VALID_MASK = ADR_STATE_VALID
    TDCP_INVALID_MASK = ADR_STATE_RESET | ADR_STATE_CYCLE_SLIP

    # Signal Thresholds
    CN0_THRESHOLD_DBHZ = 20.0  # Minimum signal strength to consider
    MAX_VELOCITY_MPS = 100.0  # Sanity check for outlier velocity

    # -------------------------------------------------------------------------
    # 4. Feature Engineering Configuration
    # -------------------------------------------------------------------------
    # The model predicts ENU (East, North, Up) residuals relative to the WLS baseline.
    # We focus on East and North for the competition metric.

    TARGET_COLUMNS = ["res_E", "res_N"]

    # Features for LightGBM
    # Includes:
    # - Signal Quality Stats (Cn0)
    # - Satellite Geometry (Elevation, Azimuth)
    # - Clock Bias/Drift uncertainty
    # - "Force" Vectors: Geometry-Projected Residuals (The core of the idea)
    # - IMU context (Acceleration/Gyro magnitude)
    FEATURE_COLUMNS = [
        # Signal Strength
        "Cn0DbHz_mean",
        "Cn0DbHz_std",
        "Cn0DbHz_min",
        "Cn0DbHz_max",
        # Satellite Counts
        "Svid_count",
        # Satellite Geometry
        "SvElevationDegrees_mean",
        "SvElevationDegrees_std",
        # Clock States
        "BiasUncertaintyNanos_mean",
        "DriftUncertaintyNanosPerSecond_mean",
        # Projected Residual Forces (Physics-Informed Features)
        # F_e, F_n: The components of the residual vector projected onto local ENU
        "Force_E",
        "Force_N",
        "Force_U",
        # Geometry Matrix Diagonals (Dilution of Precision proxies)
        "G_xx",
        "G_yy",
        "G_zz",
        # IMU Context (Dynamics)
        "Accel_Mag_mean",
        "Accel_Mag_std",
        "Gyro_Mag_mean",
        "Gyro_Mag_std",
        # Baseline Speed
        "WLS_SpeedMps",
    ]

    # -------------------------------------------------------------------------
    # 5. Model Hyperparameters (LightGBM)
    # -------------------------------------------------------------------------
    # Using MAE (L1) objective to be robust against heavy-tailed GNSS outliers.
    LGBM_PARAMS = {
        "objective": "regression_l1",  # Mean Absolute Error
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 10000,  # Large number, controlled by early stopping
        "learning_rate": 0.05,
        "num_leaves": 128,  # Slightly deeper trees for complex signal interactions
        "max_depth": -1,
        "min_child_samples": 50,  # Regularization
        "subsample": 0.8,  # Row subsampling
        "colsample_bytree": 0.8,  # Feature subsampling
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 0.1,  # L2 regularization
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Training Loop Settings
    NUM_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 150
    VERBOSE_EVAL = 100

    # -------------------------------------------------------------------------
    # 6. Post-Processing / Optimization (TDCP Alignment)
    # -------------------------------------------------------------------------
    # Optimization Objective:
    # min sum( |x_t - x_pred_t| ) + lambda * sum( ||(x_t - x_{t-1}) - v_tdcp_t * dt||^2 )

    # Weight for the TDCP shape constraint relative to the ML anchor constraint.
    # Higher value = trust physics (shape) more. Lower value = trust ML (absolute position) more.
    TDCP_LAMBDA = 5.0

    # Minimum number of satellites with valid carrier phase to compute a TDCP displacement
    TDCP_MIN_SATS = 4
