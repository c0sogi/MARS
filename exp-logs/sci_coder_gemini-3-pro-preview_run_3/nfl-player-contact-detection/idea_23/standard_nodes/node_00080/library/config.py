import os


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_23"

    # Ensure working directory exists for caching
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Data Paths
    TRACKING_PATH_TRAIN = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TRACKING_PATH_TEST = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    HELMETS_PATH_TRAIN = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    HELMETS_PATH_TEST = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    # Submission
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    N_JOBS = 12

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================

    # Temporal Lags
    # System Energy / Dynamics: Exponential Pyramid (t, t+/-1... t+/-15)
    LAGS_ENERGY = [0, -1, 1, -2, 2, -4, 4, -8, 8, -15, 15]

    # Visual Consensus: Sparse Lags (t, t+/-4, t+/-8, t+/-15)
    LAGS_VISUAL = [0, -4, 4, -8, 8, -15, 15]

    # Stream A: Interaction (Player vs Player)
    # Focus: System Energy, Relative Geometry, Visual Consensus
    STREAM_A_FEATURES = {
        "tracking_base": ["speed", "acceleration"],  # Applied to both players
        "geometry": ["distance", "relative_speed", "closure_rate"],
        "visual_consensus": ["max_iou", "min_iou", "iou_diff"],
    }

    # Stream B: Impact (Player vs Ground)
    # Focus: Strict Biomechanical Invariance (Finite-Difference Ego-Dynamics)
    # Excludes: x, y, direction, orientation, visual features
    STREAM_B_FEATURES = {
        "scalars": ["speed", "acceleration"],
        "ego_dynamics": [
            "v_surge",
            "v_sway",
            "ego_acc_surge",
            "ego_acc_sway",
            "ego_jerk_surge",
            "ego_jerk_sway",
        ],
    }

    # =========================================================================
    # Training Configuration
    # =========================================================================

    # Targeted Majority Undersampling
    # Keep 100% Positives, Subsample Negatives to 10:1 ratio
    UNDERSAMPLE_RATIO = 10.0

    # XGBoost Common Parameters
    XGB_COMMON_PARAMS = {
        "n_estimators": 5000,
        "learning_rate": 0.05,
        "tree_method": "gpu_hist",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": SEED,
        "n_jobs": N_JOBS,
    }

    # Stream A Specifics (Interaction)
    # Standard depth to model complex geometry/visual interactions
    XGB_PARAMS_STREAM_A = XGB_COMMON_PARAMS.copy()
    XGB_PARAMS_STREAM_A.update(
        {
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        }
    )

    # Stream B Specifics (Impact)
    # Robust/Invariant model: Higher regularization, stricter column sampling
    # to force reliance on ego-dynamics rather than overfitting noise.
    XGB_PARAMS_STREAM_B = XGB_COMMON_PARAMS.copy()
    XGB_PARAMS_STREAM_B.update(
        {
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.5,
            "reg_alpha": 1.0,
            "reg_lambda": 5.0,
        }
    )

    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 50
