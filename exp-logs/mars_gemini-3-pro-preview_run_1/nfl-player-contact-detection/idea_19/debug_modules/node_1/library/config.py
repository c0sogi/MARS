import os
import numpy as np


class Config:
    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    EXPERIMENT_ID = "idea_19"

    # =========================================================================
    # Directory & File Paths
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXPERIMENT_ID)

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Tracking data
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Output paths
    SUBMISSION_PATH = "./submission/submission.csv"

    # Cache paths for intermediate data (Parquet/NPY)
    CACHE_PATHS = {
        "train_features": os.path.join(WORKING_DIR, "train_features.parquet"),
        "val_features": os.path.join(WORKING_DIR, "val_features.parquet"),
        "test_features": os.path.join(WORKING_DIR, "test_features.parquet"),
        "hard_negatives": os.path.join(WORKING_DIR, "hard_negative_indices.npy"),
        "scout_preds": os.path.join(WORKING_DIR, "scout_predictions.npy"),
    }

    # =========================================================================
    # Quadratic Reachability Gating
    # =========================================================================
    # Logic: d(t) = d0 + v*t + 0.5*a*t^2
    # We solve for min(d(t)) in the window [0, GATING_WINDOW_SECONDS]
    GATING = {
        "ENABLED": True,
        "WINDOW_SECONDS": 1.0,  # Look ahead 1.0s
        "DISTANCE_THRESHOLD": 1.0,  # Yards. Keep if min_dist < 1.0
        "G_DISTANCE_SENTINEL": -1.0,  # Sentinel value for ground contact distance
    }

    # =========================================================================
    # Spectral-Kinematic Feature Engineering
    # =========================================================================
    FEATURES = {
        "WINDOW_SIZE_STEPS": 10,  # +/- 10 steps (1.0s total context)
        "USE_SPECTRAL": True,  # Enable transient spectral energy features
        "SPECTRAL_SIGMA": 1.0,  # Sigma for high-pass filtering (Gaussian diff)
        "DROP_COLS": [  # Columns to exclude to prevent overfitting/leakage
            "game_play",
            "contact_id",
            "datetime",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
            "timestamp",
            "frame",
        ],
    }

    # =========================================================================
    # Dual-Scout Diversity Mining
    # =========================================================================
    MINING = {
        "SCOUT_THRESHOLD": 0.05,  # Probability threshold to consider a negative "Hard"
        "NEG_POS_RATIO": 1.0,  # Ratio of random negatives to positives in Scout training
        "BUFFER_RATIO": 0.5,  # Ratio of random negatives to add to Expert training
    }

    # =========================================================================
    # Model Hyperparameters (Tri-Model Expert Ensemble)
    # =========================================================================

    # 1. LightGBM Expert
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "num_leaves": 256,  # High capacity
        "max_depth": 10,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "is_unbalance": True,  # Handle class imbalance
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # 2. XGBoost Expert
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 2000,
        "learning_rate": 0.02,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "enable_categorical": False,  # Features are numerical
        "random_state": SEED,
        "n_jobs": -1,
        # scale_pos_weight will be set dynamically based on training data balance
    }

    # 3. CatBoost Expert
    CATBOOST_PARAMS = {
        "iterations": 2000,
        "learning_rate": 0.02,
        "depth": 10,
        "l2_leaf_reg": 3,
        "loss_function": "Logloss",
        "eval_metric": "MCC",
        "auto_class_weights": "Balanced",
        "random_seed": SEED,
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": -1,
    }

    # =========================================================================
    # Training Loop Settings
    # =========================================================================
    TRAINING = {
        "EARLY_STOPPING_ROUNDS": 50,
        "VERBOSE_EVAL": 100,
        "USE_SAMPLE": False,  # Set to True for debugging with smaller dataset
        "SAMPLE_SIZE": 10000,
    }

    @classmethod
    def get_working_dir(cls):
        return cls.WORKING_DIR

    @classmethod
    def get_cache_path(cls, key):
        return cls.CACHE_PATHS.get(key)
