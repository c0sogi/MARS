import os


class Config:
    # ==========================================
    # PATHS & DIRECTORIES
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Ensure writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files (Pre-generated)
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Files
    TRAIN_TRACKING = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output File
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet/NPY)
    # These paths are used by the processing module to store/load deterministic data
    CACHE_TRAIN_A_X = os.path.join(WORKING_DIR, "train_streamA_X.parquet")
    CACHE_TRAIN_A_Y = os.path.join(WORKING_DIR, "train_streamA_y.npy")
    CACHE_VAL_A_X = os.path.join(WORKING_DIR, "val_streamA_X.parquet")
    CACHE_VAL_A_Y = os.path.join(WORKING_DIR, "val_streamA_y.npy")

    CACHE_TRAIN_B_X = os.path.join(WORKING_DIR, "train_streamB_X.parquet")
    CACHE_TRAIN_B_Y = os.path.join(WORKING_DIR, "train_streamB_y.npy")
    CACHE_VAL_B_X = os.path.join(WORKING_DIR, "val_streamB_X.parquet")
    CACHE_VAL_B_Y = os.path.join(WORKING_DIR, "val_streamB_y.npy")

    # ==========================================
    # HYPERPARAMETERS
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing

    # Temporal Window Configuration (Multi-Resolution)
    # Micro: High temporal resolution (flattened features)
    MICRO_WINDOW_SIZE = 4  # +/- 4 frames (9 frames total: t-4 ... t ... t+4)
    # Macro: Contextual resolution (aggregated statistics)
    MACRO_WINDOW_SIZE = 15  # +/- 15 frames (31 frames total: t-15 ... t ... t+15)

    # Data Sampling
    UNDERSAMPLE_RATIO = 10.0  # Ratio of Negative samples to Positive samples (10:1)

    # ==========================================
    # FEATURE CONFIGURATION
    # ==========================================

    # --- Stream A: Player-Player Interaction ---
    # Micro Features: Will be flattened over the window (e.g., dist_t-4, dist_t-3, ...)
    STREAM_A_MICRO_COLS = [
        "distance",
        "speed_p1",
        "speed_p2",
        "acceleration_p1",
        "acceleration_p2",
        "sin_orient_p1",
        "cos_orient_p1",
        "sin_dir_p1",
        "cos_dir_p1",
        "sin_orient_p2",
        "cos_orient_p2",
        "sin_dir_p2",
        "cos_dir_p2",
        "rel_speed",
        "closure_rate",
        "cos_sim_dir",
        "rel_acceleration",
        "inv_ttc",
    ]

    # Macro Features: Will be aggregated (mean, max, std, etc.) over the window
    STREAM_A_MACRO_COLS = [
        "distance",
        "rel_speed",
        "closure_rate",
        "speed_p1",
        "speed_p2",
        "rel_acceleration",
        "inv_ttc",
    ]

    # --- Stream B: Player-Ground Impact ---
    # Micro Features: Focus on single player kinematics and loss of control
    STREAM_B_MICRO_COLS = [
        "speed",
        "acceleration",
        "jerk",
        "sin_orient",
        "cos_orient",
        "sin_dir",
        "cos_dir",
        "angular_velocity",
        "pose_motion_align",
    ]

    # Macro Features: Contextual instability
    STREAM_B_MACRO_COLS = [
        "speed",
        "acceleration",
        "jerk",
        "angular_velocity",
        "pose_motion_align",
    ]

    # ==========================================
    # MODEL CONFIGURATION
    # ==========================================
    # XGBoost parameters optimized for A100 GPU
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.05,
        "n_estimators": 5000,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "early_stopping_rounds": 100,
        "n_jobs": 12,
        "random_state": SEED,
    }
