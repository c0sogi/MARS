import os


class Config:
    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 30: Orthogonal-Spectral Vector-Anchored Ensemble
    WORKING_DIR = "./working/idea_30"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Input Data Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Caching Paths (Parameter-Aware Caching Strategy)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Parquet files for processed features
    PROCESSED_TRAIN_PATH = os.path.join(CACHE_DIR, "processed_train.parquet")
    PROCESSED_VAL_PATH = os.path.join(CACHE_DIR, "processed_val.parquet")
    PROCESSED_TEST_PATH = os.path.join(CACHE_DIR, "processed_test.parquet")

    # Mining artifacts
    HARD_NEGATIVE_INDICES_PATH = os.path.join(CACHE_DIR, "hard_negative_indices.npy")

    # Model Artifacts
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Final Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Global Constants & Reproducibility
    # =========================================================================
    SEED = 42
    N_JOBS = 12

    # =========================================================================
    # Feature Engineering: Dual-Basis DCT & Gating
    # =========================================================================
    # Relaxed Quadratic Gating: Keep pairs within this distance
    GATING_THRESHOLD = 3.0  # Yards

    # Sentinel Value for Ground Distance (to isolate Ground branch in trees)
    SENTINEL_DIST_VALUE = -1.0

    # Discrete Cosine Transform (DCT) Parameters
    # Window size in steps (0.1s increments). 21 steps = +/- 1.0 second context.
    DCT_WINDOW_SIZE = 21
    # Number of low-frequency coefficients to retain (Trend + Shock separation)
    DCT_K = 6

    # =========================================================================
    # Training Curriculum: Anchored Mining & Smoothing
    # =========================================================================
    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000

    # Temporal Label Smoothing (Gaussian Sigma in steps)
    LABEL_SMOOTHING_SIGMA = 1.0

    # Diversity Mining
    SCOUT_PROB_THRESHOLD = 0.05  # Probability threshold to consider a negative "Hard"
    ANCHOR_RATIO = 1.0  # Ratio of Random Easy Negatives (Anchors) to Hard Negatives

    # =========================================================================
    # Tri-Ensemble Model Hyperparameters
    # =========================================================================

    # 1. LightGBM (Leaf-wise Growth) - Best for dense numerical features
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "num_leaves": 256,  # High capacity
        "max_depth": 10,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": N_JOBS,
        "verbose": -1,
        "random_state": SEED,
        "is_unbalance": True,  # Internal rebalancing
    }

    # 2. XGBoost (Level-wise Growth) - Best for approximate splits
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": "gbtree",
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "max_depth": 10,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "scale_pos_weight": 10.0,  # Explicit rebalancing weight
    }

    # 3. CatBoost (Symmetric Trees) - Best for preventing overfitting
    CAT_PARAMS = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 10,
        "l2_leaf_reg": 3,
        "subsample": 0.8,
        "verbose": 0,
        "random_seed": SEED,
        "allow_writing_files": False,
        "auto_class_weights": "Balanced",  # Internal rebalancing
    }
