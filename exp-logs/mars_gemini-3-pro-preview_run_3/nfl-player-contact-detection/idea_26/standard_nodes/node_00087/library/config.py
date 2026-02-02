import os


class Config:
    # =========================================================================
    # Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_26"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Global Configuration
    # =========================================================================
    SEED = 42
    N_JOBS = 12
    USE_CACHE = True

    # Debugging / Dataset Size Control
    # Set to a integer (e.g., 10000) to limit training data for debugging.
    # Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # =========================================================================
    # Feature Engineering Configuration
    # =========================================================================
    # Visual Consensus Pyramids Lags
    VISUAL_LAGS = [0, 4, 8, 15]

    # Stream A: Interaction Model (Player-Player)
    # Philosophy: Robust Consistency (Low-Order Physics + Visual Consensus)
    # Features focus on spatial convergence and gating perspective artifacts.
    FEATURES_STREAM_A = [
        "distance",
        "closure_rate",
        "visual_looming_rate",
        "consistency_score",
        "speed_p1",
        "speed_p2",
        "accel_p1",
        "accel_p2",
        # Visual Pyramid Features (Consensus)
        "max_iou_t0",
        "min_iou_t0",
        "iou_diff_t0",
        "max_iou_t4",
        "min_iou_t4",
        "iou_diff_t4",
        "max_iou_t8",
        "min_iou_t8",
        "iou_diff_t8",
        "max_iou_t15",
        "min_iou_t15",
        "iou_diff_t15",
    ]

    # Stream B: Impact Model (Player-Ground)
    # Philosophy: Sensitive Dynamics (High-Order Physics / Rotational)
    # Features focus on inertial signatures of falls, excluding absolute position/orientation.
    FEATURES_STREAM_B = [
        "speed",
        "acceleration",
        "ego_accel_surge",
        "ego_accel_sway",
        "ego_jerk_surge",
        "ego_jerk_sway",
        "surge_energy",
        "sway_energy",
    ]

    # =========================================================================
    # Model Hyperparameters (XGBoost)
    # =========================================================================
    # Common Parameters
    EARLY_STOPPING_ROUNDS = 50
    NEGATIVE_SAMPLE_RATIO = 0.1  # 10:1 Negative to Positive ratio for undersampling

    # Stream A: Standard Depth (6) to model complex non-linear interactions
    XGB_PARAMS_STREAM_A = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "random_state": SEED,
        "n_jobs": N_JOBS,
    }

    # Stream B: Shallow Depth (6) + Explicit Physics to avoid overfitting sensor noise
    XGB_PARAMS_STREAM_B = {
        "n_estimators": 2000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist",
        "random_state": SEED,
        "n_jobs": N_JOBS,
    }

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories for caching and submission.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
