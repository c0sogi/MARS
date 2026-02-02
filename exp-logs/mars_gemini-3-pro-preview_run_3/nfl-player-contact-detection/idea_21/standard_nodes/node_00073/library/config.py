import os


class Config:
    """
    Global configuration for the Orthogonal-Physics Dual-Stream GBDT pipeline.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Paths
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================
    # Exponential Temporal Pyramids: Sparse lags for flattening features
    # Used for both Relational Scalars (Stream A) and Kinetic features (Stream B)
    LAGS = [0, 1, 2, 4, 8, 15]

    # =========================================================================
    # Training Configuration
    # =========================================================================
    # Targeted Majority Undersampling
    # Ratio of Negative samples to Positive samples (10:1)
    NEGATIVE_SAMPLE_RATIO = 10.0

    # Early Stopping Rounds for XGBoost
    EARLY_STOPPING_ROUNDS = 50

    # Number of steps for linear threshold optimization
    THRESHOLD_OPT_STEPS = 100

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================

    # Stream A: Interaction Model (Player-Player)
    # Optimized for Spatial Convergence using Relational Scalars & Visual Consensus.
    # Uses standard regularization (max_depth=6).
    STREAM_A_PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.02,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Stream B: Impact Model (Player-Ground)
    # Optimized for Force Transients using Finite-Difference Ego-Dynamics.
    # Uses Moderate Depth Constraint (max_depth=8) to model non-linear ego-dynamics
    # while preventing overfitting to sensor noise.
    STREAM_B_PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.02,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,  # Slightly higher to be robust against tracking noise
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "random_state": SEED,
        "n_jobs": -1,
    }
