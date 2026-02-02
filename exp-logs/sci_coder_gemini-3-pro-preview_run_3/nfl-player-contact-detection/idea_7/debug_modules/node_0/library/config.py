import os
import numpy as np


class Config:
    """
    Global configuration for the NFL Contact Detection pipeline.
    Implements the Multi-Modal Late-Fusion Ensemble strategy (Stream A: Tracking, Stream B: Helmets).
    """

    # =========================================================================
    # 1. File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate features (Parquet/NPY)
    # Using 'idea_7' to distinguish this iteration's cache
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Data Paths
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    # Metadata Paths (Pre-split Train/Val/Test)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # 2. Global Settings
    # =========================================================================
    SEED = 42
    NUM_JOBS = 12
    USE_GPU = True

    # =========================================================================
    # 3. Data Processing Configuration
    # =========================================================================
    # Temporal Window Sizes (Indices, at 10Hz)
    # Micro: Capture instantaneous mechanics (approx +/- 0.4s)
    WINDOW_MICRO = 4
    # Macro: Capture trajectory context (approx +/- 1.5s)
    WINDOW_MACRO = 15

    # Undersampling Ratio for Training (Negative Samples : Positive Samples)
    # 10:1 ratio balances efficiency with sufficient negative exposure
    NEG_POS_RATIO = 10.0

    # =========================================================================
    # 4. Feature Whitelists (Deterministic Ordering)
    # =========================================================================

    # --- Stream A: Tracking (Kinematics) ---
    # Base columns from tracking data to be used
    TRACKING_BASE_COLS = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Derived columns to generate before windowing
    TRACKING_DERIVED_COLS = [
        "sin_direction",
        "cos_direction",
        "sin_orientation",
        "cos_orientation",
    ]

    # Interaction columns (Player-Player only)
    TRACKING_INTERACTION_COLS = ["distance", "relative_speed"]

    # --- Stream B: Helmets (Visual Geometry) ---
    # Base columns from helmet data
    HELMET_BASE_COLS = ["left", "top", "width", "height"]

    # Derived columns
    HELMET_DERIVED_COLS = ["centroid_x", "centroid_y", "area"]

    # Interaction columns (Player-Player only)
    HELMET_INTERACTION_COLS = ["iou", "dist_centroids", "area_ratio"]

    # =========================================================================
    # 5. Model Hyperparameters
    # =========================================================================
    # XGBoost Configuration
    # Optimized for A100 GPU usage
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        # GPU Acceleration settings
        "device": "cuda",
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": NUM_JOBS,
        # Execution settings
        "verbose": 0,
        "verbosity": 0,
    }

    # Training settings
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # Inference / Ensemble Settings
    # Grid search ranges for blending weights and thresholds
    BLEND_WEIGHTS = np.linspace(0, 1, 21)  # 0.0, 0.05, ... 1.0
    THRESHOLDS = np.linspace(0.1, 0.9, 81)  # 0.10, 0.11, ... 0.90
