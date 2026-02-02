import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_26"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Data Configuration
# ==========================================
# List of sensor columns in the CSV files
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]

# Global Random Seed for Reproducibility
SEED = 42

# Number of Cross-Validation Folds
N_FOLDS = 5

# ==========================================
# Feature Engineering Hyperparameters
# ==========================================
# Savitzky-Golay Filter for Trend Extraction (View A)
# Large window to strictly isolate low-frequency drift
SG_WINDOW_SIZE = 51
SG_POLYORDER = 2

# Welch's Method for PSD Band Power (View C)
# High nperseg to ensure resolution in Low (0.1-3Hz) band
WELCH_NPERSEG = 1024

# Wavelet Transform for Texture Analysis (View B)
WAVELET_TYPE = "db4"

# Flattened Temporal Profiling
# Number of non-overlapping windows to divide the signal into
N_TEMPORAL_SEGMENTS = 10

# ==========================================
# Model Hyperparameters (LightGBM)
# ==========================================
# High-Capacity Regressor with L2 Loss
LGBM_PARAMS = {
    "objective": "regression_l2",  # L2 Loss (Mean Squared Error)
    "metric": "mae",  # Monitor MAE
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": 128,  # High capacity
    "learning_rate": 0.02,  # Low LR for convergence
    "n_estimators": 10000,  # Large number of trees, controlled by early stopping
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "seed": SEED,
    "deterministic": True,
}

# Training Control
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100
