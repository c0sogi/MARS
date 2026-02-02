import os


class Config:
    # =========================================================================
    # Global Constants & Paths
    # =========================================================================
    SEED = 42

    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Output/Working Directories
    # Specific working directory for Idea 23
    WORKING_DIR = "./working/idea_23"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # File Paths
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Processing & Gating Hyperparameters
    # =========================================================================
    # Sentinel Value for Ground interactions (distance set to this)
    SENTINEL_VALUE = -1.0

    # Relaxed Quadratic Gating
    # Keep pairs where min(distance) in window < GATING_DISTANCE
    GATING_DISTANCE = 3.0  # yards

    # Temporal Label Smoothing
    # Gaussian smoothing sigma for binary labels to handle +/- 10Hz noise
    LABEL_SMOOTHING_SIGMA = 1.0

    # Mining Thresholds
    # Probability threshold for a negative to be considered "Hard"
    HARD_NEGATIVE_THRESHOLD = 0.05

    # Anchor Mining
    # Ratio of random easy negatives (anchors) to keep relative to the number of hard negatives
    # This prevents model collapse by preserving the global decision boundary
    ANCHOR_RATIO = 1.0

    # =========================================================================
    # Feature Engineering
    # =========================================================================
    # Vector-Decomposed Kinematic Features
    # Explicitly distinguishing between Radial (Impact) and Tangential (Shear) components
    FEATURES = [
        "distance",
        "time_to_collision",
        "radial_velocity",  # Impact speed
        "tangential_velocity",  # Shear speed
        "radial_acceleration",
        "tangential_acceleration",
        "radial_accel_energy",  # Transient Spectral Energy (RMS of high-passed radial accel)
        "speed_p1",  # Scalar magnitude retained
        "speed_p2",
        "acceleration_p1",
        "acceleration_p2",
    ]

    # =========================================================================
    # Model Hyperparameters (Tri-Ensemble)
    # =========================================================================
    # High-capacity configuration: Deep trees (num_leaves=256, max_depth=10)

    # 1. LightGBM (Leaf-wise growth)
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 256,
        "max_depth": 10,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbosity": -1,
        "n_jobs": -1,
        "is_unbalance": True,  # Handle class imbalance internal to LGBM
        "seed": SEED,
    }

    # 2. XGBoost (Level-wise growth)
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 10,
        "max_leaves": 256,  # XGBoost 3.0+ supports this
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "enable_categorical": False,
        "scale_pos_weight": 10.0,  # Approximate imbalance handling
        "random_state": SEED,
        "verbosity": 0,
    }

    # 3. Scikit-Learn HistGradientBoosting (Symmetric/Oblivious-like behavior)
    # Replacing CatBoost as it is not in the installed packages list
    HISTGB_PARAMS = {
        "loss": "log_loss",
        "learning_rate": 0.05,
        "max_iter": 1000,
        "max_leaf_nodes": 256,
        "max_depth": 10,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 50,
        "random_state": SEED,
        "class_weight": "balanced",  # Handle imbalance
    }

    # Training settings
    NUM_BOOST_ROUND = 2000
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 100

    @classmethod
    def setup(cls):
        """Ensure working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
