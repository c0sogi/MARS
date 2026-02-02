import os

# =============================================================================
# 1. PATH CONFIGURATION
# =============================================================================

INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"

# Working directory for idea_11 (Physics-Consistency Featurized Median-Ensemble)
# This directory will store cached parquet files and model artifacts
WORKING_DIR = "./working/idea_11"
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Submission Path
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# =============================================================================
# 2. GLOBAL CONSTANTS
# =============================================================================

SEED = 42
N_FOLDS = 5  # Number of folds for GroupKFold cross-validation

# WGS84 Ellipsoid Constants (Required for ECEF <-> LLA conversions and physics calcs)
WGS84_A = 6378137.0  # Major axis (meters)
WGS84_F = 1.0 / 298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Minor axis (meters)
LIGHT_SPEED = 299792458.0  # Speed of light (m/s)
OMEGA_EARTH = 7.2921151467e-5  # Earth rotation rate (rad/s)

# =============================================================================
# 3. FEATURE CONFIGURATION
# =============================================================================

# Base GNSS features aggregated per timestamp
BASE_GNSS_FEATURES = [
    "Cn0DbHz_mean",
    "Cn0DbHz_max",
    "Cn0DbHz_std",
    "sv_count",
]

# IMU features (Magnitude of uncalibrated readings)
IMU_FEATURES = [
    "accel_mag_mean",
    "gyro_mag_mean",
]

# Physics-Consistency Features
# These features measure the quality of the baseline WLS solution by checking
# consistency with raw measurements (pseudorange and doppler).
CONSISTENCY_FEATURES = [
    # Geometric Consistency (Post-Fit Residuals): Std and MeanAbs of (Observed Pr - Calculated Pr)
    "pr_residual_mean_abs",
    "pr_residual_std",
    # Dynamic Consistency (Doppler Residuals): Std and MeanAbs of (Observed Rate - Calculated Rate)
    "doppler_residual_mean_abs",
    "doppler_residual_std",
]

# Final list of features to be used by the model
# We strictly exclude raw timestamps or absolute lat/lon to prevent overfitting to trajectory shapes
FEATURE_COLS = BASE_GNSS_FEATURES + IMU_FEATURES + CONSISTENCY_FEATURES

# Target columns for residual learning (East, North offsets in meters)
TARGET_COLS = ["delta_east", "delta_north"]

# =============================================================================
# 4. MODEL HYPERPARAMETERS
# =============================================================================

# LightGBM parameters optimized for MAE (L1 loss) and robustness against heavy-tailed outliers
LGBM_PARAMS = {
    "objective": "regression_l1",  # L1 loss (MAE) is crucial for this task
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 5000,  # High cap, controlled by early stopping
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,  # Silent execution
}

# Training loop settings
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100  # Print metrics every 100 rounds
