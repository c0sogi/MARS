import os

# -----------------------------------------------------------------------------
# 1. Global Configuration & Flags
# -----------------------------------------------------------------------------
SEED = 42
DEBUG = False  # Set to True to run on a small subset of data for debugging
DEBUG_SAMPLE_SIZE = 500  # Number of drives/phones to sample in debug mode

# -----------------------------------------------------------------------------
# 2. Directories and File Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache Paths (Parquet files for processed features)
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# Model Save Paths
MODEL_EAST_PATH = os.path.join(WORKING_DIR, "lgbm_east.txt")
MODEL_NORTH_PATH = os.path.join(WORKING_DIR, "lgbm_north.txt")

# Final Submission Path
FINAL_SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# 3. Physical Constants (WGS84)
# -----------------------------------------------------------------------------
# Used for Geodetic <-> ECEF <-> ENU transformations
WGS84_A = 6378137.0  # Semi-major axis (meters)
WGS84_F = 1 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis (meters)
WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)  # First eccentricity squared

# -----------------------------------------------------------------------------
# 4. Feature Engineering Configuration
# -----------------------------------------------------------------------------
# GNSS Columns to load
GNSS_COLS = [
    "utcTimeMillis",
    "Cn0DbHz",
    "SvElevationDegrees",
    "Svid",
    "SignalType",
    "RawPseudorangeMeters",
    "SvPositionXEcefMeters",
    "SvPositionYEcefMeters",
    "SvPositionZEcefMeters",
    "WlsPositionXEcefMeters",
    "WlsPositionYEcefMeters",
    "WlsPositionZEcefMeters",
]

# IMU Columns to load
IMU_COLS = [
    "utcTimeMillis",
    "MeasurementX",
    "MeasurementY",
    "MeasurementZ",
    "MessageType",
]

# Features to generate per timestamp
FEATURE_NAMES = [
    "Cn0DbHz_mean",
    "Cn0DbHz_max",
    "Cn0DbHz_std",
    "Svid_count",
    "SvElevationDegrees_mean",
    "Accel_Mag_mean",
    "Accel_Mag_std",
]

# -----------------------------------------------------------------------------
# 5. Model Hyperparameters (LightGBM Quantile Regression)
# -----------------------------------------------------------------------------
# Quantiles to predict: Lower bound, Median (Correction), Upper bound
QUANTILES = [0.1, 0.5, 0.9]

# LightGBM Parameters
LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "quantile",
    "metric": "quantile",
    "n_estimators": 5000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
}

# Training Loop Settings
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100

# -----------------------------------------------------------------------------
# 6. Kalman Smoothing Parameters
# -----------------------------------------------------------------------------
# Process Noise (Q) - Uncertainty in motion model (constant velocity)
PROCESS_NOISE_STD = 0.5  # meters/second^2

# Measurement Noise Scaling
# The measurement noise R will be dynamic: R_t = (Uncertainty_t * SCALE_FACTOR) + BASE
UNCERTAINTY_SCALE_FACTOR = 1.0
BASE_MEASUREMENT_NOISE = 2.0  # Minimum noise floor in meters
