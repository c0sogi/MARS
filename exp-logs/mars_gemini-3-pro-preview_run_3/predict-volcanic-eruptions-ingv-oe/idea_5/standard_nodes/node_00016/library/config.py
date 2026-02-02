import os

# ==========================================
# Global Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Constants
# ==========================================
RANDOM_SEED = 42
SAMPLING_RATE = 100  # 60000 samples / 600 seconds (10 mins) = 100 Hz
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
TARGET_COL = "time_to_eruption"
ID_COL = "segment_id"

# ==========================================
# Feature Engineering Hyperparameters
# ==========================================

# 1. Temporal Segmentation (Explicit Windowing)
# We split the 10-minute segment into N non-overlapping windows to capture
# the trajectory of the signal (e.g., is energy increasing?).
N_TIME_SEGMENTS = 10  # 10 segments of 1 minute each

# 2. Rolling Statistics
# Window size for rolling calculations (e.g., Rolling RMS)
# 100 samples = 1 second
ROLLING_WINDOW_SIZE = 100

# 3. Spectral Analysis (Band Power)
# Nyquist frequency is 50Hz. We define physically relevant bands.
FREQ_BANDS = {
    "low": (0.1, 2.0),  # Long-period tremors
    "mid": (2.0, 10.0),  # Intermediate frequency
    "high": (10.0, 45.0),  # High frequency noise/events
}

# 4. Cepstral Analysis (MFCC)
# Parameters for extracting Mel-Frequency Cepstral Coefficients
MFCC_PARAMS = {
    "n_mfcc": 13,  # Number of coefficients to keep
    "n_fft": 2048,  # FFT window size
    "hop_length": 512,  # Hop length between frames
    "n_mels": 128,  # Number of Mel bands
}

# ==========================================
# Feature Selection (RFE) Configuration
# ==========================================
# Recursive Feature Elimination parameters to reduce dimensionality
# and prevent overfitting.
RFE_PARAMS = {
    "n_features_to_select": 50,  # Strict limit on feature count
    "step": 0.05,  # Remove 5% of features per iteration
    "verbose": 0,
    # Random Forest parameters used for the RFE estimator
    "estimator_params": {
        "n_estimators": 50,
        "max_depth": 10,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    },
}
# Fraction of training data to use for fitting RFE (to save compute time)
RFE_TRAIN_SUBSET_SIZE = 0.3

# ==========================================
# Model Hyperparameters (LightGBM)
# ==========================================
LGBM_PARAMS = {
    "objective": "regression_l1",  # Mean Absolute Error
    "metric": "mae",
    "n_estimators": 10000,  # High cap, controlled by early stopping
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.7,  # Subsample features per tree
    "bagging_fraction": 0.7,  # Subsample data per iteration
    "bagging_freq": 1,
    "lambda_l1": 0.5,  # L1 Regularization
    "lambda_l2": 0.5,  # L2 Regularization
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
}

# Training control
EARLY_STOPPING_ROUNDS = 200
VERBOSE_EVAL = 200

# ==========================================
# Debug / Runtime Flags
# ==========================================
# If True, runs on a small subset of data for quick pipeline verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 100
