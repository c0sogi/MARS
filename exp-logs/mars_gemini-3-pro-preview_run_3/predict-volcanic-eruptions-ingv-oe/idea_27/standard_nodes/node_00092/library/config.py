import os
import numpy as np

# ==========================================
# Global Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_28"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Reproducibility and Debugging
# ==========================================
SEED = 42
DEBUG = False  # Set to True to run on a small subset of data
DEBUG_SAMPLE_SIZE = 200  # Number of samples to use if DEBUG is True

# ==========================================
# Data Characteristics
# ==========================================
NUM_SENSORS = 10
SIGNAL_LENGTH = 60001
SAMPLING_RATE = 100  # Hz (60001 samples / 600 seconds)

# ==========================================
# Signal Processing Parameters
# ==========================================
# Savitzky-Golay Filter (View A: Trend)
SG_WINDOW = 51
SG_POLYORDER = 2

# Discrete Wavelet Transform (View B: Texture)
DWT_WAVELET = "db4"

# Power Spectral Density (View C: High-Res Spectral)
PSD_NPERSEG = 1024  # High window size for better low-freq resolution
PSD_BANDS = {"low": (0.1, 3), "mid": (3, 10), "high": (10, 45)}

# Temporal Flattening (View C: Temporal Profiling)
# Number of non-overlapping windows to divide the signal into
TEMPORAL_WINDOW_COUNT = 10

# ==========================================
# Feature Engineering Configuration
# ==========================================
# List of sensor columns in the raw CSVs
SENSOR_COLS = [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]

# ==========================================
# Model Hyperparameters (LightGBM)
# ==========================================
# High-Capacity Homogeneous Ensemble settings
N_FOLDS = 5

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "l2",  # Mean Squared Error
    "boosting_type": "gbdt",
    "num_leaves": 128,  # High capacity
    "learning_rate": 0.02,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 10000,
    "early_stopping_rounds": 100,
    "verbosity": -1,
    "n_jobs": -1,
    "seed": SEED,
    "force_col_wise": True,
}
