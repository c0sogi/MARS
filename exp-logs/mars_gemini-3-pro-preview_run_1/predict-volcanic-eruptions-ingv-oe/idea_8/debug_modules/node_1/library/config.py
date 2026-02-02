import os
import numpy as np

# ==========================================
# 1. PATHS & DIRECTORIES
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache Paths (Parquet/NPY)
TRAIN_TABULAR_PATH = os.path.join(WORKING_DIR, "train_tabular_features.parquet")
VAL_TABULAR_PATH = os.path.join(WORKING_DIR, "val_tabular_features.parquet")
TEST_TABULAR_PATH = os.path.join(WORKING_DIR, "test_tabular_features.parquet")

TRAIN_SPECTROGRAMS_PATH = os.path.join(WORKING_DIR, "train_spectrograms.npy")
VAL_SPECTROGRAMS_PATH = os.path.join(WORKING_DIR, "val_spectrograms.npy")
TEST_SPECTROGRAMS_PATH = os.path.join(WORKING_DIR, "test_spectrograms.npy")

# Model Save Paths
LGBM_MODEL_DIR = os.path.join(WORKING_DIR, "lgbm_models")
CNN_MODEL_DIR = os.path.join(WORKING_DIR, "cnn_models")
META_MODEL_PATH = os.path.join(WORKING_DIR, "meta_ridge_model.joblib")

os.makedirs(LGBM_MODEL_DIR, exist_ok=True)
os.makedirs(CNN_MODEL_DIR, exist_ok=True)

# ==========================================
# 2. GLOBAL CONFIGURATION
# ==========================================
SEED = 42
N_FOLDS = 5
NUM_WORKERS = 4  # For data loading

# Sensor Definitions
SENSORS = [f"sensor_{i}" for i in range(1, 11)]
NUM_SENSORS = 10
SEGMENT_LENGTH = 60001  # Fixed length based on data analysis

# ==========================================
# 3. TABULAR FEATURE CONFIGURATION (Branch A)
# ==========================================
# Beamforming: Create a virtual channel by averaging all normalized sensors
USE_BEAMFORMING = True

# MFCC Configuration
# Restrict to coefficients 1-13 (Low-Order) to capture spectral envelope
MFCC_PARAMS = {
    "n_mfcc": 13,  # Number of coefficients to keep
    "n_fft": 2048,
    "hop_length": 512,
    "sr": 100,  # Approx sampling rate (60k samples / 600s)
}

# Feature Aggregation Stats
# Strictly limited to robust statistics to avoid outlier artifacts
# Excludes Min/Max for MFCCs
ROBUST_STATS_LIST = ["mean", "std", "q05", "q95"]
GLOBAL_STATS_LIST = [
    "mean",
    "std",
    "skew",
    "kurtosis",
    "q01",
    "q05",
    "q95",
    "q99",
    "abs_q95",
    "abs_q99",
]

# ==========================================
# 4. VISION MODEL CONFIGURATION (Branch B)
# ==========================================
CNN_CONFIG = {
    "model_name": "efficientnet_b0",
    "in_channels": 10,  # 1 channel per sensor
    "img_size": (128, 128),  # Time x Frequency dimensions for spectrogram resize
    "batch_size": 32,
    "epochs": 35,  # Extended training for convergence
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "early_stopping_patience": 5,
    "use_log_target": True,  # Apply np.log1p to target for training
    "instance_norm": True,  # Apply (x - mean) / std per sample
}

# Spectrogram Generation Params
SPECTROGRAM_PARAMS = {
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 128,
    "fmin": 0,
    "fmax": 50,  # Nyquist is 50Hz given sr=100
}

# ==========================================
# 5. LIGHTGBM CONFIGURATION (Branch A)
# ==========================================
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "n_estimators": 5000,
    "early_stopping_round": 100,
    "verbose": -1,
    "n_jobs": -1,
    "seed": SEED,
}

# ==========================================
# 6. META-LEARNER CONFIGURATION
# ==========================================
RIDGE_PARAMS = {"alpha": 1.0, "fit_intercept": True, "random_state": SEED}
