import os

# ==========================================
# Path Configuration
# ==========================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Feature Cache Paths (Parquet format)
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")

# Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")


# ==========================================
# Global Constants
# ==========================================

SEED = 42
NUM_SENSORS = 10
SENSOR_COLS = [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]

# Debugging / Runtime Control
# Set to an integer (e.g., 100) to limit the number of files processed for debugging.
# Set to None to process the full dataset.
DEBUG_SAMPLE_SIZE = None


# ==========================================
# Signal Processing Configuration
# ==========================================

# 1. Imputation
FILL_NA_STRATEGY = "mean"  # Impute NaNs with column mean per segment

# 2. Savitzky-Golay Filter (Smoothing & Derivatives)
# Used to smooth signal before calculating kinematic features (velocity/acceleration)
SG_WINDOW_LENGTH = 21  # Must be an odd integer
SG_POLYORDER = 2  # Polynomial order

# 3. Wavelet Decomposition
# Discrete Wavelet Transform parameters for multi-resolution analysis
WAVELET_NAME = "db4"  # Daubechies 4
WAVELET_LEVELS = 4  # Number of decomposition levels

# 4. Temporal Windowing
# Number of non-overlapping windows to split the signal into for temporal evolution stats
NUM_WINDOWS = 10


# ==========================================
# Model Hyperparameters (LightGBM)
# ==========================================

# Training Control
N_ESTIMATORS = 10000
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100

# Model Parameters
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "n_estimators": N_ESTIMATORS,
    "learning_rate": 0.01,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_jobs": -1,
    "seed": SEED,
    "force_col_wise": True,
}
