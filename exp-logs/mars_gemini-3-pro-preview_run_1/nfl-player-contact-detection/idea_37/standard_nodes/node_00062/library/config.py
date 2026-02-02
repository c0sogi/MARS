import os


class Config:
    # =========================================================================
    # DIRECTORIES & PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for idea_37 caching
    WORKING_DIR = "./working/idea_37"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working and submission directories exist
    # Note: In a pure config file, side effects are sometimes avoided,
    # but ensuring paths exist is practical here.
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Tracking Data Paths
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Baseline Helmets (if needed for auxiliary features)
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    SEED = 42
    N_JOBS = 12  # 12 vCPUs available

    # =========================================================================
    # FEATURE ENGINEERING & GATING
    # =========================================================================
    # Window size: +/- 10 frames (approx +/- 1.0s window total around step)
    # Used for Dynamic Basis calculation and Trajectory Projection
    FEATURE_WINDOW_SIZE = 10

    # Relaxed Quadratic Gating
    # Filter pairs where min_distance in window < GATING_THRESHOLD
    GATING_THRESHOLD = 3.0  # Yards

    # Sentinel value for distance when calculating Player-Ground interactions
    GROUND_DISTANCE_SENTINEL = -1.0

    # Feature Flags
    USE_DYNAMIC_BASIS = True
    USE_DECOUPLED_KINEMATICS = True

    # =========================================================================
    # TRAINING STRATEGY (ANCHORED MINING)
    # =========================================================================
    # Ratio of Random Easy Negatives (Anchors) to Positives in the Expert Set
    ANCHOR_RATIO = 1.0

    # Probability threshold to consider a negative as "Hard" during Scout phase
    HARD_NEGATIVE_THRESHOLD = 0.05

    # =========================================================================
    # MODEL HYPERPARAMETERS
    # =========================================================================
    # Common settings
    N_ESTIMATORS = 2000
    EARLY_STOPPING_ROUNDS = 50

    # LightGBM Parameters (Leaf-wise growth)
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 256,  # Deep trees for high capacity
        "max_depth": 10,  # Constraint to prevent overfitting
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": N_ESTIMATORS,
        "is_unbalance": True,  # Handle imbalance internally
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "verbose": -1,
    }

    # XGBoost Parameters (Level-wise growth)
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": N_ESTIMATORS,
        # Approx ratio for Positives : (Anchors + Hard Negs) ~ 1:3
        "scale_pos_weight": 3.0,
        "tree_method": "hist",  # Efficient histogram-based training
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "enable_categorical": False,
    }

    # Ensemble Weights (Unweighted Average)
    ENSEMBLE_WEIGHTS = {"lgbm": 0.5, "xgb": 0.5}

    # =========================================================================
    # DEBUGGING & RUNTIME
    # =========================================================================
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False

    # Sample size for debugging (number of unique plays or interactions)
    DEBUG_SAMPLE_SIZE = 5000
