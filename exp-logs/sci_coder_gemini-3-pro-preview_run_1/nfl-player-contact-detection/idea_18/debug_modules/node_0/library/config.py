import os
import numpy as np


class Config:
    # =========================================================================
    # Global System Configuration
    # =========================================================================
    SEED = 42
    N_CPU = 12

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Tracking Data
    TRACKING_PATH_TRAIN = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TRACKING_PATH_TEST = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Working Directory for Caching and Models (Idea 18)
    WORKING_DIR = "./working/idea_18"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Stage 0: Quadratic Reachability Gating Configuration
    # =========================================================================
    # Logic: d(t) = d0 + v*t + 0.5*a*t^2
    # We look ahead to find if min(d(t)) < threshold within the window
    GATING = {
        "ENABLED": True,
        "LOOKAHEAD_STEPS": 10,  # 1.0 seconds (10 steps @ 10Hz)
        "DIST_THRESHOLD": 1.5,  # Yards. Slightly generous to avoid false negatives.
        "TIME_STEP": 0.1,  # Seconds per step
    }

    # =========================================================================
    # Stage 1: Spectral-Kinematic Feature Engineering Configuration
    # =========================================================================
    FEATURES = {
        "WINDOW_SIZE": 10,  # +/- 10 steps for feature calculation (Total 2.1s window)
        "USE_SPECTRAL": True,  # Enable Transient Spectral Energy features
        "SPECTRAL_SMOOTHING": 3,  # Window size for rolling mean to separate Trend vs Shock
        "SENTINEL_VALUE": -1.0,  # Distance value for Ground interactions
        "DROP_COLS": [
            "game_play",
            "game_key",
            "play_id",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "step",
            "datetime",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
            "contact_id",
        ],
    }

    # =========================================================================
    # Training Configuration
    # =========================================================================
    TRAINING = {
        "SCOUT_EPOCHS": 1000,
        "EXPERT_EPOCHS": 3000,
        "EARLY_STOPPING_ROUNDS": 50,
        "VERBOSE_EVAL": 100,
        "SCOUT_MINING_THRESHOLD": 0.05,  # Probability threshold to consider a negative "Hard"
        "NEGATIVE_SAMPLING_RATIO": 1.0,  # Ratio of random negatives to positives in Scout training
        "EXPERT_NEGATIVE_BUFFER": 0.5,  # Additional random negatives ratio for Expert training
    }

    # =========================================================================
    # Model Hyperparameters (Unified Heterogeneous Expert Ensemble)
    # =========================================================================

    # 1. LightGBM (Leaf-wise)
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "average_precision",  # Optimize for ranking/precision
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 256,  # High capacity
        "max_depth": 10,  # Deep trees
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "is_unbalance": True,  # Handle class imbalance
        "verbosity": -1,
        "n_jobs": N_CPU,
        "seed": SEED,
    }

    # 2. XGBoost (Level-wise)
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.02,
        "max_depth": 10,  # Deep trees
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",  # Efficient histogram-based training
        "scale_pos_weight": 10.0,  # Approx imbalance ratio handling (tuned)
        "n_jobs": N_CPU,
        "random_state": SEED,
        "enable_categorical": False,  # We handle categoricals via encoding if any
    }

    # 3. CatBoost (Symmetric / Oblivious Trees)
    # Note: CatBoost params are typically passed to the constructor or fit method
    CATBOOST_PARAMS = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",  # CatBoost likes AUC/Logloss
        "learning_rate": 0.02,
        "depth": 10,  # Deep trees
        "l2_leaf_reg": 3,
        "auto_class_weights": "Balanced",  # Handle imbalance
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "verbose": 100,
        "allow_writing_files": False,
        "thread_count": N_CPU,
        "random_seed": SEED,
    }
