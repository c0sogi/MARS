import os
import numpy as np


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea iteration to ensure safe caching
    WORKING_DIR = "./working/idea_4"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Tracking Data Paths
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Baseline Helmets Paths (if needed for future expansion, though not primary for this idea)
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    # Submission Path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================
    # Window size for temporal flattening.
    # +/- 10 steps results in a total window of 21 frames (Current + 10 past + 10 future)
    WINDOW_HALF_SIZE = 10

    # Flags for specific feature sets
    USE_SPATIAL_DENSITY = True  # Context: Count of players within 1.5 yards
    USE_IMPACT_PHYSICS = True  # Context: Jerk and Angular Jerk derivatives

    # Caching Filenames (Base names, hash to be appended in processing script)
    CACHE_TRAIN_FEATURES = "train_features_context"
    CACHE_VAL_FEATURES = "val_features_context"
    CACHE_TEST_FEATURES = "test_features_context"

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================

    # LightGBM Configuration
    # High capacity (num_leaves=256) to absorb massive dataset.
    # is_unbalance=True handles the extreme class imbalance internally.
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",  # optimizing logloss, monitoring MCC manually
        "boosting_type": "gbdt",
        "num_leaves": 256,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "is_unbalance": True,
        "verbosity": -1,
        "random_state": SEED,
        "n_jobs": 12,
    }

    # XGBoost Configuration
    # High capacity (max_depth=10).
    # scale_pos_weight is set to None here; it must be calculated dynamically
    # as (neg_count / pos_count) during training.
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 10,
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda",  # Use GPU
        "random_state": SEED,
        "n_jobs": 12,
    }

    # Training Loop Settings
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 50
