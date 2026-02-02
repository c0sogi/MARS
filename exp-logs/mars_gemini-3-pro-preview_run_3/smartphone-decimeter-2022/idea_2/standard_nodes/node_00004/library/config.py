import os

# ==========================================
# Directories and File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Constants
# ==========================================
SEED = 42
WGS84_A = 6378137.0  # Semi-major axis
WGS84_B = 6356752.314245  # Semi-minor axis

# ==========================================
# Data Loading Configuration
# ==========================================
# Columns to load from device_gnss.csv
GNSS_COLS = [
    "utcTimeMillis",
    "TimeNanos",
    "FullBiasNanos",
    "BiasNanos",
    "Svid",
    "Cn0DbHz",
    "SvElevationDegrees",
    "SvAzimuthDegrees",
    "WlsPositionXEcefMeters",
    "WlsPositionYEcefMeters",
    "WlsPositionZEcefMeters",
]

# Columns to load from device_imu.csv
IMU_COLS = [
    "utcTimeMillis",
    "MessageType",
    "MeasurementX",
    "MeasurementY",
    "MeasurementZ",
]

# ==========================================
# Feature Engineering Configuration
# ==========================================
# Aggregated features to be generated
AGG_FEATURES = [
    "Cn0DbHz_mean",
    "Cn0DbHz_std",
    "Cn0DbHz_max",
    "Cn0DbHz_min",
    "SvElevationDegrees_mean",
    "SvElevationDegrees_std",
    "sv_count",
    "imu_acc_mag_mean",
    "imu_acc_mag_std",
]

# Targets for the regression models
TARGETS = ["lat_error", "lon_error"]

# ==========================================
# Model Hyperparameters (LightGBM)
# ==========================================
LGBM_PARAMS = {
    "objective": "regression_l1",  # Mean Absolute Error (L1)
    "metric": "mae",
    "boosting_type": "gbdt",
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
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Training Loop Settings
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100
