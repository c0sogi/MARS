import os


class Config:
    # =========================================================================
    # Global Configuration
    # =========================================================================
    SEED = 42

    # =========================================================================
    # Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Input File Paths
    # =========================================================================
    # Tracking Data
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Helmet Data (Visual-Geometric Stream)
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")

    # Labels and Submission
    TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Generated Splits)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # =========================================================================
    # Feature Engineering Parameters
    # =========================================================================
    # Temporal Window Sizes (in 0.1s steps)
    # Micro: Capture immediate motion context (+/- 0.4s)
    MICRO_WINDOW_SIZE = 4
    # Macro: Capture broader play context (+/- 1.5s)
    MACRO_WINDOW_SIZE = 15

    # Sampling
    # Negative to Positive ratio for Random Undersampling
    UNDERSAMPLE_RATIO = 10.0

    # =========================================================================
    # Model Hyperparameters (XGBoost)
    # =========================================================================
    # Common parameters for both Stream A (Kinematic) and Stream B (Visual)
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
        "device": "cuda",  # Use GPU acceleration
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Blending Configuration
    # Number of trials for optimizing the blending weights
    BLENDING_TRIALS = 50
