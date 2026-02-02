import os
import numpy as np

# =============================================================================
# 1. Global Paths & Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_21"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Sample Submission
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "dataset_train.parquet")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "dataset_val.parquet")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "dataset_test.parquet")

# =============================================================================
# 2. Physical Constants
# =============================================================================
# Speed of light in m/s
LIGHT_SPEED = 299792458.0

# GNSS Carrier Frequencies (Hz)
L1_FREQ = 1575.42e6
L5_FREQ = 1176.45e6

# Earth Rotation Rate (rad/s)
OMEGA_E = 7.2921151467e-5

# WGS84 Ellipsoid Constants
WGS84_A = 6378137.0  # Semi-major axis
WGS84_F = 1.0 / 298.257223563  # Flattening

# =============================================================================
# 3. Data Processing Configuration
# =============================================================================
# Random Seed for reproducibility
SEED = 42

# Flag to load cached feature engineering results
# Set to True to speed up subsequent runs if data is already processed
LOAD_CACHED_DATA = True

# Debugging: Limit number of rows/drives to process
# Set to None for full run
DEBUG_ROWS = None

# =============================================================================
# 4. Model Hyperparameters (LightGBM)
# =============================================================================
LGBM_PARAMS = {
    "objective": "mae",  # Mean Absolute Error for robustness to outliers
    "boosting_type": "gbdt",
    "n_estimators": 2000,  # Maximum number of trees
    "learning_rate": 0.05,  # Learning rate
    "num_leaves": 128,  # Complexity of trees
    "colsample_bytree": 0.8,  # Feature subsampling
    "subsample": 0.8,  # Row subsampling
    "subsample_freq": 1,
    "reg_alpha": 0.1,  # L1 regularization
    "reg_lambda": 0.1,  # L2 regularization
    "random_state": SEED,
    "n_jobs": -1,  # Use all available CPUs
    "verbose": -1,  # Suppress LightGBM output
}

# Early stopping rounds for training
EARLY_STOPPING_ROUNDS = 100

# =============================================================================
# 5. Adaptive Graph Optimization Parameters
# =============================================================================
# Weights for the Factor Graph Cost Function
# J(x) = Sum(Huber(x - ML)) + Sum(Weight * ||(x_t - x_{t-1}) - Delta_Kin||^2)

# Weight for the ML Anchor prediction term
WEIGHT_ANCHOR = 1.0

# Weight for Time-Differenced Carrier Phase (TDCP) constraints
# TDCP is extremely precise (mm-level), so this should be very high relative to anchors
WEIGHT_TDCP = 100.0

# Weight for Doppler-derived velocity constraints
# Doppler is less precise than TDCP but robust, used when TDCP fails
WEIGHT_DOPPLER = 10.0

# Huber Loss Delta for the Anchor term
# Errors larger than this (in meters) will be treated linearly instead of quadratically
# This allows the trajectory to break away from bad ML predictions
HUBER_DELTA = 5.0

# RANSAC Parameters for Kinematic Estimation
RANSAC_THRESHOLD_METERS = 0.5  # Threshold for inlier detection in velocity estimation
RANSAC_MIN_SAMPLES = 4  # Minimum satellites required
