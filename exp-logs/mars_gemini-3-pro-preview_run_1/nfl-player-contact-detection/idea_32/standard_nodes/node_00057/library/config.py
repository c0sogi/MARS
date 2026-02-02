import os


class Config:
    # =========================================================================
    # DIRECTORIES AND PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_32"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet/NPY)
    # Using specific filenames for the idea_32 pipeline
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")

    CACHE_HARD_NEGATIVES = os.path.join(WORKING_DIR, "hard_negative_indices.npy")
    CACHE_BEST_THRESHOLD = os.path.join(WORKING_DIR, "best_threshold.npy")

    # Model Artifacts
    # Scouts
    MODEL_SCOUT_LGBM = os.path.join(WORKING_DIR, "scout_lgbm.joblib")
    MODEL_SCOUT_XGB = os.path.join(WORKING_DIR, "scout_xgb.joblib")
    MODEL_SCOUT_CAT = os.path.join(WORKING_DIR, "scout_cat.joblib")

    # Experts
    MODEL_EXPERT_LGBM = os.path.join(WORKING_DIR, "expert_lgbm.joblib")
    MODEL_EXPERT_XGB = os.path.join(WORKING_DIR, "expert_xgb.joblib")
    MODEL_EXPERT_CAT = os.path.join(WORKING_DIR, "expert_cat.joblib")

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    SEED = 42
    N_JOBS = 12  # Number of vCPUs available

    # Debugging flag to use smaller dataset
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50000

    # =========================================================================
    # FEATURE ENGINEERING PARAMETERS
    # =========================================================================
    # Window size for Time-Domain features (steps before/after current step)
    # Step is 0.1s. +/- 10 steps = +/- 1.0 second context.
    WINDOW_SIZE = 10

    # Sentinel value for Ground distance (non-physical value to isolate Ground interactions)
    GROUND_DISTANCE_SENTINEL = -1.0

    # Quadratic Gating Threshold (yards)
    # Pairs with min distance > this value in the window are discarded early
    GATING_DISTANCE_THRESHOLD = 3.0

    # =========================================================================
    # TRAINING HYPERPARAMETERS
    # =========================================================================
    # Hard Negative Mining
    SCOUT_PREDICT_THRESHOLD = (
        0.05  # Probability threshold to consider a negative as "Hard"
    )
    ANCHOR_RATIO = (
        1.0  # Ratio of Random Easy Negatives to Hard Negatives in Expert Dataset
    )

    # LightGBM Parameters (Deep Trees, Leaf-wise)
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 256,
        "max_depth": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
        "is_unbalance": True,  # Internal rebalancing
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": N_JOBS,
    }

    # XGBoost Parameters (Deep Trees, Level-wise)
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 10,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
        "scale_pos_weight": 10,  # Approximate imbalance handling
        "verbosity": 0,
        "seed": SEED,
        "n_jobs": N_JOBS,
        "tree_method": "hist",
    }

    # CatBoost Parameters (Symmetric Trees)
    # Note: Requires catboost package. If not installed, this config will simply not be used by the runner.
    CAT_PARAMS = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 10,
        "l2_leaf_reg": 3,
        "random_seed": SEED,
        "early_stopping_rounds": 50,
        "auto_class_weights": "Balanced",  # Internal rebalancing
        "verbose": 0,
        "thread_count": N_JOBS,
        "allow_writing_files": False,
    }
