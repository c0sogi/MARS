import os


class Config:
    """
    Configuration for the Invariant-Physics Temporal Pyramid Dual-Stream GBDT.
    Centralizes paths, feature definitions, and model hyperparameters.
    """

    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # Ensure necessary write directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Files
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    N_JOBS = 12
    USE_GPU = True

    # Debugging / Development Controls
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000  # Rows to process if DEBUG is True

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================
    # Exponential Temporal Pyramids: Lags to flatten
    LAGS = [1, 2, 4, 8, 15]

    # -------------------------------------------------------------------------
    # Stream A: Interaction Model (Player vs Player)
    # -------------------------------------------------------------------------
    # Features to be computed, lagged, and flattened for Stream A.
    # Focus: Relational dynamics, System Energy, and Visual Consensus.
    STREAM_A_FEATURES = [
        # Relational Scalars
        "distance",  # Euclidean distance
        "closure_rate",  # Finite difference of distance (t - t-1)
        # System Energy (Absolute Kinematics)
        "speed_p1",
        "speed_p2",
        "acceleration_p1",
        "acceleration_p2",
        # Visual Consensus (IoU metrics)
        "sideline_iou",
        "endzone_iou",
        "max_iou",
        "min_iou",
        "iou_diff",  # abs(sideline - endzone)
        # Cross-Modal Consistency
        "looming_mismatch",  # diff between kinematic closure and visual looming
        # Relative Geometry (Cite Lesson 74)
        "cos_sim_dir",
        "cos_sim_orient",
    ]

    # -------------------------------------------------------------------------
    # Stream B: Impact Model (Player vs Ground)
    # -------------------------------------------------------------------------
    # Features to be computed, lagged, and flattened for Stream B.
    # Focus: Strictly Invariant Ego-Centric Kinematics.
    STREAM_B_FEATURES = [
        # Invariant Baseline
        "speed",
        "acceleration",  # Scalar magnitude
        # Ego-Centric Kinematics (Projected onto Body Orientation)
        "v_surge",  # Velocity component along orientation
        "v_sway",  # Velocity component orthogonal to orientation
        "a_surge",  # Acceleration component along orientation
        "a_sway",  # Acceleration component orthogonal to orientation
        # Higher-Order Derivatives (Cite Lesson 27, 34, 70)
        "jerk",
        "angular_velocity",
        "j_surge",
        "j_sway",
    ]

    # Explicitly excluded features to prevent overfitting to field location
    STREAM_B_EXCLUDE = [
        "x_position",
        "y_position",
        "direction",
        "orientation",
        "jersey_number",
        "team",
    ]

    # =========================================================================
    # Training Configuration
    # =========================================================================
    # Targeted Majority Undersampling (Negative : Positive ratio)
    UNDERSAMPLE_RATIO = 10.0

    # XGBoost Hyperparameters
    # Stream A: Deep Trees for Complex Interaction Manifolds
    XGB_PARAMS_A = {
        "n_estimators": 2500,
        "learning_rate": 0.02,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist" if USE_GPU else "hist",
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "enable_categorical": False,
    }

    # Stream B: Shallower Trees for Robust, Invariant Dynamics
    XGB_PARAMS_B = {
        "n_estimators": 2500,
        "learning_rate": 0.02,
        "max_depth": 7,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist" if USE_GPU else "hist",
        "random_state": SEED,
        "n_jobs": N_JOBS,
        "enable_categorical": False,
    }

    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100
