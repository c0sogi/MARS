import os


class Config:
    """
    Configuration for the Stacked Heterogeneous Ensemble with Robust Quantile-Kinematic Features.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Specifications
    # ==========================================
    # 10 Sensor columns
    SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]

    # Sampling rate estimation: 60,000 rows over 10 minutes (600 seconds) = 100 Hz
    SAMPLING_RATE = 100

    # ==========================================
    # Feature Engineering Hyperparameters
    # ==========================================
    # 1. Robust Smoothing (Savitzky-Golay)
    # Window size must be odd and > 20 as per requirements
    SAVGOL_WINDOW = 25
    SAVGOL_POLYORDER = 2

    # 2. Statistical Features (Robust Quantiles)
    # Explicitly prioritizing quantiles over unstable moments (skew/kurtosis)
    QUANTILES = [0.01, 0.05, 0.95, 0.99]

    # 4. Windowing (Flattened Robust Windows)
    # Split 60,000 rows into N non-overlapping windows
    N_WINDOWS = 10

    # 5. Spectral Bands (Hz)
    # Based on 100Hz sampling rate (Nyquist = 50Hz)
    FREQ_BANDS = {
        "low": (0.1, 2.0),
        "mid": (2.0, 10.0),
        "high": (10.0, 20.0),
        "ultra": (20.0, 45.0),
    }

    # ==========================================
    # Model Hyperparameters (Level 0: Base Learners)
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    EARLY_STOPPING_ROUNDS = 200
    VERBOSE_EVAL = 100

    # LightGBM Params
    LGBM_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "random_state": SEED,
        "n_jobs": -1,
        "device": "gpu",
        "verbose": -1,
    }

    # XGBoost Params
    XGB_PARAMS = {
        "objective": "reg:absoluteerror",
        "eval_metric": "mae",
        "n_estimators": 10000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "gpu_hist",  # GPU acceleration
        "random_state": SEED,
        "n_jobs": -1,
    }

    # CatBoost Params
    CAT_PARAMS = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": 10000,
        "learning_rate": 0.01,
        "depth": 6,
        "task_type": "GPU",  # GPU acceleration
        "verbose": 0,
        "random_seed": SEED,
    }

    # ==========================================
    # Model Hyperparameters (Level 1: Meta Learner)
    # ==========================================
    # Ridge Regression Alpha (Regularization strength)
    RIDGE_ALPHA = 10.0

    # ==========================================
    # Compute Resources
    # ==========================================
    NUM_WORKERS = 12
