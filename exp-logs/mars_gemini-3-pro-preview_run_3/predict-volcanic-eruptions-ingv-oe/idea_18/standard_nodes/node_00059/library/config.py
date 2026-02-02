import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Using idea_18 directory for caching as per the task context
WORKING_DIR = "./working/idea_18"
SUBMISSION_DIR = "./submission"

# Ensure necessary writeable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# Data Processing Configuration
# ==========================================
# Seismic data sampling rate (implied from 60001 rows in 10 mins)
SAMPLE_RATE = 100

# Savitzky-Golay Filter Parameters (View A: Trend)
# Used to isolate low-frequency kinematic motion
# Idea specifies window > 20, e.g., 51
SAVGOL_WINDOW = 51
SAVGOL_POLY = 3

# Spectral Analysis Parameters (View C: Spectral)
# Explicit Closed Intervals for Band Power
FREQ_BANDS = [
    (0.1, 3.0),  # Low Frequency
    (3.0, 10.0),  # Mid Frequency
    (10.0, 45.0),  # High Frequency
]

# Temporal Shift-Invariance Parameters (View C: Temporal)
# Number of non-overlapping windows to divide the signal into
N_TEMPORAL_WINDOWS = 10

# ==========================================
# Feature Engineering Configuration
# ==========================================
# List of sensors available in the dataset
SENSORS = [f"sensor_{i}" for i in range(1, 11)]

# ==========================================
# Model Configuration (LightGBM)
# ==========================================
# Single, Highly-Optimized LightGBM Regressor
# Optimized for MAE (regression_l1) with high capacity (num_leaves=128)
LGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "boosting_type": "gbdt",
    "n_estimators": 6000,
    "learning_rate": 0.01,
    "num_leaves": 128,
    "max_depth": -1,
    "min_child_samples": 50,  # min_data_in_leaf
    "reg_alpha": 1.0,  # Lambda L1
    "reg_lambda": 1.0,  # Lambda L2
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_jobs": -1,
    "verbose": -1,
    "random_state": SEED,
}

# ==========================================
# Training Configuration
# ==========================================
N_FOLDS = 5
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100

# ==========================================
# Debug / Development Configuration
# ==========================================
# Set DEBUG to True to run on a smaller subset of data for testing
DEBUG = False
DEBUG_SAMPLE_SIZE = 500
