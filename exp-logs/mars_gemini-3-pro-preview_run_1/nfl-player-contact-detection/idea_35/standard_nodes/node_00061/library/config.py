import os


class Config:
    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    EXP_NAME = "idea_35"

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", EXP_NAME)

    # Ensure working directory exists for caching
    os.makedirs(WORKING_DIR, exist_ok=True)

    # =========================================================================
    # File Paths
    # =========================================================================
    # Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Tracking Data
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_OUTPUT_PATH = "submission.csv"

    # =========================================================================
    # Caching Paths (Parameter-Aware)
    # =========================================================================
    # Parquet files for feature sets
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "features_train.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "features_val.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "features_test.parquet")

    # Numpy files for indices
    CACHE_HARD_NEGATIVES = os.path.join(WORKING_DIR, "hard_negative_indices.npy")

    # Model Artifacts
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # =========================================================================
    # Feature Engineering Hyperparameters
    # =========================================================================
    # Window size for Dynamic Interaction-Aligned Basis Projections
    # Window is [-WINDOW_SIZE, +WINDOW_SIZE] relative to current step
    WINDOW_SIZE = 10

    # Relaxed Quadratic Reachability Gating
    # Maximum predicted distance allowed to retain a pair
    GATING_THRESHOLD = 3.0  # Yards

    # Sentinel Value for Ground Distance
    GROUND_DISTANCE_SENTINEL = -1.0

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Diversity Mining
    HARD_NEGATIVE_THRESHOLD = (
        0.05  # Probability threshold to consider a negative "Hard"
    )

    # Anchored Expert Training
    ANCHOR_RATIO = 1.0  # Ratio of Random Easy Negatives to Positives (1:1)

    # Optimization
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 50

    # =========================================================================
    # Model Hyperparameters (Unified Heterogeneous Tri-Ensemble)
    # =========================================================================

    # 1. LightGBM (Leaf-wise growth)
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "num_leaves": 256,
        "max_depth": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "min_data_in_leaf": 20,
        "verbose": -1,
        "n_jobs": -1,
        "seed": SEED,
        "is_unbalance": True,  # Internal rebalancing
    }

    # 2. XGBoost (Level-wise growth)
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": SEED,
        # scale_pos_weight is often tuned dynamically, but setting a default high value
        # helps with the initial imbalance before mining.
        "scale_pos_weight": 10.0,
    }

    # 3. CatBoost (Symmetric trees)
    CATBOOST_PARAMS = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "learning_rate": 0.05,
        "iterations": 2000,
        "depth": 10,
        "l2_leaf_reg": 3,
        "random_seed": SEED,
        "verbose": 0,
        "allow_writing_files": False,
        "auto_class_weights": "Balanced",
    }
