import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
WORKING_DIR = os.path.join(BASE_DIR, "working")

# Specific directory for Idea 17 (Band Power + Rich Kinematics)
# This is where cached feature engineering results (parquet files) will be stored.
CACHE_DIR = os.path.join(WORKING_DIR, "idea_17")
os.makedirs(CACHE_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
NUM_SENSORS = 10
SENSOR_COLS = [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]
TARGET_COL = "time_to_eruption"

# =============================================================================
# SIGNAL PROCESSING HYPERPARAMETERS
# =============================================================================
# Savitzky-Golay Filter for Trend Extraction (View A)
# Window length must be odd.
# Assuming ~100Hz sampling (60k rows/10mins), 51 is approx 0.5 seconds.
# This window size is chosen to smooth out noise while retaining kinematic trends.
SAVGOL_WINDOW_LENGTH = 51
SAVGOL_POLYORDER = 2

# Shift-Invariant Temporal Statistics
# Number of non-overlapping windows to divide the signal into before aggregating stats.
# This allows capturing signal volatility without overfitting to specific time indices.
N_TEMPORAL_WINDOWS = 10

# Wavelet for Texture Extraction (View B)
WAVELET_NAME = "db4"

# =============================================================================
# MODEL HYPERPARAMETERS (LightGBM)
# =============================================================================
# Training settings
N_ESTIMATORS = 6000
EARLY_STOPPING_ROUNDS = 200
VERBOSE_EVAL = 500

# Model parameters optimized for MAE
LGBM_PARAMS = {
    "objective": "regression_l1",  # L1 regression optimizes MAE directly
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,  # Low learning rate for stability with high estimators
    "num_leaves": 63,  # Higher complexity to capture interactions in rich feature set
    "max_depth": -1,
    "min_data_in_leaf": 50,  # Regularization to prevent overfitting on outliers
    "feature_fraction": 0.8,  # Subsample features per tree
    "bagging_fraction": 0.8,  # Subsample data per iteration
    "bagging_freq": 1,
    "lambda_l1": 1.5,  # L1 Regularization
    "lambda_l2": 1.0,  # L2 Regularization
    "n_jobs": -1,
    "random_state": SEED,
    "verbosity": -1,
}
