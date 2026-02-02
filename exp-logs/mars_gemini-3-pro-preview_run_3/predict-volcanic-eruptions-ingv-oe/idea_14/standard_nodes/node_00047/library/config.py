import os


class Config:
    """
    Configuration class for the Multi-View Dual-Stream Stacking strategy.
    Defines paths, signal processing parameters, and model hyperparameters.
    """

    # ==========================================
    # Directory and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_optimized"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Global Configuration
    # ==========================================
    SEED = 42
    N_JOBS = 12  # Utilizing available vCPUs

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data for rapid iteration
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200

    # ==========================================
    # Feature Engineering / Signal Processing
    # ==========================================
    # Stream B: Smoothed Kinematics
    # Window size must be odd and > 20 to prevent noise amplification
    SG_WINDOW_SIZE = 31
    SG_POLY_ORDER = 2

    # View 3: Explicit Temporal Evolution
    NUM_TEMPORAL_WINDOWS = 10

    # View 4: Structural Spectral Texture
    WAVELET_NAME = "db4"

    # General Data
    NUM_SENSORS = 10

    # ==========================================
    # Model Hyperparameters (Level 0: Base Learners)
    # ==========================================
    # Shared Training Parameters
    N_FOLDS = 5
    N_ESTIMATORS = 10000
    EARLY_STOPPING_ROUNDS = 100

    # LightGBM Configuration
    # Robust performance on CPU with high core count
    LGBM_PARAMS = {
        "n_estimators": N_ESTIMATORS,
        "learning_rate": 0.01,
        "num_leaves": 31,
        "max_depth": -1,
        "objective": "regression_l1",  # Optimize for MAE
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    # XGBoost Configuration
    # Leveraging A100 GPU for acceleration
    XGB_PARAMS = {
        "n_estimators": N_ESTIMATORS,
        "learning_rate": 0.01,
        "max_depth": 6,
        "objective": "reg:absoluteerror",  # Optimize for MAE
        "eval_metric": "mae",
        "tree_method": "hist",
        "device": "cuda",  # GPU
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "verbosity": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # CatBoost Configuration
    # Leveraging A100 GPU for acceleration
    CATBOOST_PARAMS = {
        "iterations": N_ESTIMATORS,
        "learning_rate": 0.01,
        "depth": 6,
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "task_type": "GPU",  # GPU
        "random_seed": SEED,
        "verbose": 0,
        "allow_writing_files": False,
    }

    # ==========================================
    # Model Hyperparameters (Level 1: Meta Learner)
    # ==========================================
    # Ridge Regression for Stacking
    # Linear meta-learner to combine OOF predictions
    RIDGE_PARAMS = {"alpha": 1.0, "random_state": SEED, "fit_intercept": True}
