import os
import numpy as np
import random


class Config:
    # =========================================================================
    # Global Settings & Reproducibility
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 10000  # Number of rows to sample in debug mode

    # =========================================================================
    # Directory & File Paths
    # =========================================================================
    # Base directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Files (Parquet format)
    TRAIN_FEATURES_CACHE = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_CACHE = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_CACHE = os.path.join(WORKING_DIR, "test_features.parquet")
    MODEL_PATH = os.path.join(WORKING_DIR, "lgbm_model.txt")

    # =========================================================================
    # Feature Engineering Hyperparameters
    # =========================================================================
    # Temporal Window: Number of frames before and after the current step to include
    # 59.94 Hz video, 10 Hz tracking.
    # Window of +/- 10 steps covers +/- 1.0 second.
    WINDOW_PRE = 10
    WINDOW_POST = 10

    # Columns to use from tracking data for feature construction
    TRACKING_COLS = [
        "x_position",
        "y_position",
        "speed",
        "acceleration",
        "direction",
        "orientation",
    ]

    # Derived features to compute per timestep before flattening
    DERIVED_COLS = ["distance", "speed_diff", "acc_diff"]

    # =========================================================================
    # Model Hyperparameters (LightGBM)
    # =========================================================================
    # Using 'is_unbalance': True to handle the severe class imbalance (approx 1:72)
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 2000,  # High number, controlled by early stopping
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "is_unbalance": True,  # Key for handling rare contact events
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Training settings
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 50

    # =========================================================================
    # Setup Methods
    # =========================================================================
    @classmethod
    def setup(cls):
        """
        Initialize directories and random seeds.
        """
        # Create working and submission directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds for reproducibility
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        os.environ["PYTHONHASHSEED"] = str(cls.SEED)
