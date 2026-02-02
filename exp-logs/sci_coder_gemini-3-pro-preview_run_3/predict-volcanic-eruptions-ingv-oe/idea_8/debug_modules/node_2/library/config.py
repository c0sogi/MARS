import os

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Global Constants
# ==========================================
SEED = 42
N_FOLDS = 5
NUM_SENSORS = 10
SENSOR_COLS = [f"sensor_{i}" for i in range(1, NUM_SENSORS + 1)]

# ==========================================
# Signal Processing & Feature Extraction
# ==========================================
# Savitzky-Golay Filter settings for robust smoothing
SAVGOL_WINDOW = 25  # Must be > 20 per strategy
SAVGOL_POLYORDER = 3

# Quantiles for kinematic features (explicitly excluding min/max/skew/kurtosis)
QUANTILES = [0.01, 0.05, 0.95, 0.99]

# Wavelet settings
WAVELET_TYPE = "db4"

# Windowing for "Flattened Robust Windows"
# Splitting 10-minute segment into sub-windows
NUM_WINDOWS = 10

# ==========================================
# Model Hyperparameters
# ==========================================

# Level 0: Base Learners

# LightGBM
LGBM_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "num_leaves": 64,
    "max_depth": -1,
    "objective": "regression",
    "metric": "mae",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# XGBoost
# Utilizing GPU (A100)
XGB_PARAMS = {
    "n_estimators": 5000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "device": "cuda",  # Use GPU
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": SEED,
    "n_jobs": -1,
}

# HistGradientBoostingRegressor (Scikit-Learn)
# Used as a proxy for CatBoost given installed package constraints
HGB_PARAMS = {
    "max_iter": 2000,
    "learning_rate": 0.05,
    "max_leaf_nodes": 64,
    "max_depth": None,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "loss": "absolute_error",
    "random_state": SEED,
    "verbose": 0,
}

# Level 1: Meta Learner

# Ridge Regression
RIDGE_PARAMS = {"alpha": 10.0, "fit_intercept": True, "random_state": SEED}

# ==========================================
# Training Settings
# ==========================================
EARLY_STOPPING_ROUNDS = 100
VERBOSE_EVAL = 100
