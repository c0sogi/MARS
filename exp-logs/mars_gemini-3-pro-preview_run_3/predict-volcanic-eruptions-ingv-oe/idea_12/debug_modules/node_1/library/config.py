import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_12"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
# This satisfies the requirement to ensure directories exist for caching/saving
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Global Configuration
# ==========================================
SEED = 42
N_FOLDS = 5
NUM_WORKERS = 12  # Utilizing the 12 vCPUs available
DEBUG = False  # Toggle for rapid prototyping on smaller subsets

# ==========================================
# Signal Processing Configuration
# ==========================================
# Savitzky-Golay Filter Settings (Stream B)
# Window size must be odd and > 20 as per "Idea 7" and "Idea 12" logic
SG_WINDOW = 21
SG_POLY = 3

# Wavelet Transform Settings (View 3)
WAVELET_TYPE = "db4"

# Temporal Windowing (View 5)
# Input signal length is ~60,001.
# Dividing into 60 windows gives ~1000 samples (10 seconds) per window.
N_TEMPORAL_WINDOWS = 60

# ==========================================
# Model Hyperparameters
# ==========================================

# LightGBM Hyperparameters (Base Learner 1)
# Optimized for Mean Absolute Error (MAE)
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "n_estimators": 10000,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "random_state": SEED,
    "n_jobs": -1,
    "early_stopping_rounds": 100,
}

# XGBoost Hyperparameters (Base Learner 2)
# Configured for GPU acceleration (A100)
XGB_PARAMS = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "n_estimators": 10000,
    "learning_rate": 0.01,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "device": "cuda",  # XGBoost 3.0+ syntax for GPU
    "random_state": SEED,
    "n_jobs": -1,
    "early_stopping_rounds": 100,
}

# CatBoost Hyperparameters (Base Learner 3)
# Configured for GPU acceleration
CATBOOST_PARAMS = {
    "loss_function": "MAE",
    "iterations": 10000,
    "learning_rate": 0.01,
    "depth": 6,
    "task_type": "GPU",
    "verbose": 0,
    "random_seed": SEED,
    "early_stopping_rounds": 100,
}

# Ridge Regression Hyperparameters (Meta Learner)
# Linear stacker for Level 1
RIDGE_PARAMS = {"alpha": 10.0, "random_state": SEED}
