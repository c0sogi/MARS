import os


class Config:
    """
    Global configuration for the Biomechanical Dual-Stream GBDT pipeline.
    Handles file paths, feature engineering constants, and model hyperparameters.
    """

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # File Paths
    # =========================================================================
    # Metadata (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Sources
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    N_JOBS = 12  # Utilizing all 12 vCPUs

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================
    # Sparse Temporal Lags for Exponential Temporal Pyramids
    # Captures immediate physics (t+/-1, 2) and trajectory context (t+/-4, 8, 15)
    LAGS = [-15, -8, -4, -2, -1, 0, 1, 2, 4, 8, 15]

    # Sentinel value for missing visual data (e.g., when helmet not detected)
    MISSING_VAL = -999

    # =========================================================================
    # Model Configuration (XGBoost)
    # =========================================================================
    # Common parameters for GPU acceleration
    COMMON_XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",  # Leverage NVIDIA A100
        "predictor": "gpu_predictor",
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    # Stream A: Interaction Model (Player-Player)
    # Features: Field-Centric Relational + Ego-Centric Relational + Visual
    STREAM_A_PARAMS = COMMON_XGB_PARAMS.copy()
    STREAM_A_PARAMS.update(
        {
            "learning_rate": 0.05,
            "n_estimators": 3000,
            "max_depth": 8,  # Standard depth for relational features
            "min_child_weight": 10,
        }
    )

    # Stream B: Impact Model (Player-Ground)
    # Features: Field-Centric Kinematics + Ego-Centric Self-Motion
    # Note: Reduced depth to prevent overfitting on noisy kinematic derivatives (Cite solution_lesson_node_00062)
    STREAM_B_PARAMS = COMMON_XGB_PARAMS.copy()
    STREAM_B_PARAMS.update(
        {
            "learning_rate": 0.05,
            "n_estimators": 3000,
            "max_depth": 6,
            "min_child_weight": 5,
        }
    )

    # Training Loop Settings
    EARLY_STOPPING_ROUNDS = 50

    # Data Sampling
    # Random Undersampling Ratio (Negative : Positive)
    NEG_POS_RATIO = 10.0
