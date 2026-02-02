import os


class Config:
    # =========================================================================
    # Global Configuration
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a smaller subset of data
    NUM_WORKERS = 4

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching intermediate files (parquet/npy)
    # Using 'idea_22' as specified for this specific architecture iteration
    WORKING_DIR = "./working/idea_22"
    SUBMISSION_DIR = "./submission"

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # File Paths
    # =========================================================================
    # Raw Data
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    TRAIN_VIDEO_META = os.path.join(INPUT_DIR, "train_video_metadata.csv")
    TEST_VIDEO_META = os.path.join(INPUT_DIR, "test_video_metadata.csv")

    # Metadata (Splits)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================

    # Temporal Window (Lags)
    # We use a sparse window to capture dynamics without exploding feature count
    # 0 is current step. +/- steps at 10Hz (e.g., 15 steps = 1.5 seconds)
    LAG_OFFSETS = [-15, -8, -4, -2, -1, 0, 1, 2, 4, 8, 15]

    # -------------------------------------------------------------------------
    # Stream A: The Collider (Player-Player Interaction)
    # -------------------------------------------------------------------------
    # Philosophy: Maximal Context & Energy Awareness.
    # Includes absolute kinematics, relative geometry, and visual cues.

    STREAM_A_FEATURES = [
        # 1. Relational Scalars (Proximity & Convergence)
        "distance",  # Euclidean distance
        "rel_speed",  # Magnitude of relative velocity vector
        "closure_rate",  # Derivative of distance (negative = closing)
        # 2. Absolute Kinematics (System Energy)
        "speed_p1",
        "speed_p2",
        "acceleration_p1",
        "acceleration_p2",  # Scalar magnitude
        # 3. Ego-Relational Projections (Angle of Attack)
        # Project relative velocity onto P1's orientation
        "rel_surge",  # Closing speed from front/back
        "rel_sway",  # Closing speed from side
        # 4. Visual Pyramids (Looming & Consensus)
        "iou_sideline",
        "iou_endzone",
        "dist_sideline",
        "dist_endzone",
        "iou_diff",  # |Sideline - Endzone| (Uncertainty)
    ]

    # -------------------------------------------------------------------------
    # Stream B: The Accelerometer (Player-Ground Impact)
    # -------------------------------------------------------------------------
    # Philosophy: Strict Biomechanical Invariance.
    # Excludes x, y, direction, orientation, and visual features.
    # Focuses on internal forces and shock.

    STREAM_B_FEATURES = [
        # 1. Finite-Difference Ego-Dynamics
        "surge",  # Velocity projected onto orientation
        "sway",  # Velocity projected orthogonal to orientation
        "ego_acc_surge",  # Derivative of surge velocity
        "ego_acc_sway",  # Derivative of sway velocity
        "ego_jerk_surge",  # Derivative of ego-acceleration (Shock)
        "ego_jerk_sway",  # Derivative of ego-acceleration (Shock)
        # 2. Invariant Scalar Kinematics
        "speed",  # Absolute speed magnitude
        "acceleration",  # Absolute acceleration magnitude
        "sa",  # Signed acceleration (provided in tracking)
    ]

    # =========================================================================
    # Training Configuration
    # =========================================================================

    # Undersampling
    # Retain 100% of positives, subsample negatives to this ratio (Neg:Pos)
    NEG_POS_RATIO = 10.0

    # Model Hyperparameters (XGBoost)
    # Using 'hist' tree method for GPU acceleration

    # Stream A Hyperparameters (Complex Interaction)
    # Standard depth to model complex interactions between visual/spatial/temporal
    XGB_PARAMS_STREAM_A = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Stream B Hyperparameters (Simple Physics)
    # Shallow depth to prevent overfitting to noise, relying on strong 'Jerk' signals
    XGB_PARAMS_STREAM_B = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,  # Kept constrained
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,  # Higher regularization
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": SEED,
        "n_jobs": -1,
    }

    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    # Threshold optimization search space
    THRESHOLD_START = 0.1
    THRESHOLD_END = 0.6
    THRESHOLD_STEP = 0.01
