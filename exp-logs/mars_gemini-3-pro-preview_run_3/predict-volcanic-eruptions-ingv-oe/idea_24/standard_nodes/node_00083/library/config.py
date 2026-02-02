import os
import numpy as np

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata paths
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching features (Idea 25 specific)
WORKING_DIR = "./working/idea_25_optimized"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission output
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
SEED = 42
N_FOLDS = 5
N_JOBS = 12  # Utilize available vCPUs

# =============================================================================
# DATA SPECIFICATIONS
# =============================================================================
SAMPLING_RATE = 100  # Hz (60001 samples / 600 seconds)
NUM_SENSORS = 10
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]

# =============================================================================
# SIGNAL PROCESSING HYPERPARAMETERS
# =============================================================================
# 1. Trend Extraction (Savitzky-Golay)
# Large window to capture low-frequency baseline drift
SG_WINDOW = 51
SG_POLYORDER = 2

# 2. Texture Analysis (Wavelet Transform)
WAVELET_TYPE = "db4"

# 3. Hierarchical Temporal Aggregation
# Split signal into N windows, compute stats per window, then aggregate those stats
HIERARCHICAL_WINDOWS = 10

# 4. Spectral Analysis (PSD Band Power via Welch's Method)
# Frequency bands in Hz
PSD_BANDS = {"low": (0.1, 3.0), "mid": (3.0, 10.0), "high": (10.0, 20.0)}


# =============================================================================
# MODEL HYPERPARAMETERS (High-Capacity LightGBM)
# =============================================================================
def get_lgbm_params(overrides=None):
    """
    Returns the dictionary of LightGBM parameters.
    Allows for dynamic overrides for debugging or tuning.
    """
    params = {
        "objective": "regression_l2",  # MSE Loss as per Idea 24
        "metric": "mae",  # Monitor MAE
        "boosting_type": "gbdt",
        "n_estimators": 10000,  # High capacity
        "learning_rate": 0.01,  # Lower learning rate for better generalization (Cite solution_lesson_node_00064)
        "num_leaves": 128,  # Deep trees for complex interactions
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbosity": -1,
        "n_jobs": N_JOBS,
        "seed": SEED,
        "early_stopping_rounds": 100,
    }

    if overrides:
        params.update(overrides)

    return params


# =============================================================================
# DEBUGGING / DEVELOPMENT
# =============================================================================
# Set to a small number (e.g., 100) to speed up pipeline development
# Set to None for full run
DEBUG_SAMPLE_SIZE = None
