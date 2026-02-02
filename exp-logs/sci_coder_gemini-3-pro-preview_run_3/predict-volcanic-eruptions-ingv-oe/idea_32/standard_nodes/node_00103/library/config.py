import os
import numpy as np

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Cache directory for Idea 33
WORKING_DIR = "./working/idea_33"
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Global Constants
# ==========================================
SEED = 42
N_FOLDS = 5
N_JOBS = 12  # Utilizing available vCPUs

# ==========================================
# Feature Engineering Configuration
# ==========================================
# Savitzky-Golay Filter (Trend Extraction)
SG_WINDOW = 51
SG_POLYORDER = 2

# Discrete Wavelet Transform (Texture Extraction)
WAVELET_TYPE = "db4"

# Welch PSD (Spectral Structure)
PSD_NPERSEG = 1024
PSD_BANDS = {"low": (0.1, 3), "mid": (3, 10), "high": (10, 45)}

# Dense Quantiles for Trend Shape Profiling
TREND_QUANTILES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

# Temporal Profiling Windows
N_TEMPORAL_WINDOWS = 10

# ==========================================
# Model Hyperparameters
# ==========================================
# High-Capacity LightGBM Configuration
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "num_leaves": 128,  # High capacity as per strategy
    "learning_rate": 0.01,  # Low LR for convergence with deep trees
    "n_estimators": 10000,  # High number of estimators
    "feature_fraction": 0.8,  # Subsampling features
    "bagging_fraction": 0.8,  # Subsampling data (Bagging)
    "bagging_freq": 1,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": SEED,
    "random_state": SEED,
}

# Training Control
EARLY_STOPPING_ROUNDS = 100

# ==========================================
# Debugging / Runtime Control
# ==========================================
# Controls the number of samples used for training/validation.
# Set to None to use the full dataset.
# Set to an integer (e.g., 500) for rapid prototyping/debugging.
DEBUG_SAMPLE_SIZE = None
