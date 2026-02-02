import os
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # 1. Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_24"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    SUBMISSION_PATH = "./submission/submission.csv"

    # Cache Paths (Parquet/NPY)
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features_vaam.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features_vaam.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features_vaam.parquet")
    CACHE_HARD_NEGATIVES = os.path.join(WORKING_DIR, "hard_negative_indices.npy")

    # -------------------------------------------------------------------------
    # 2. Global Constants & Physics Parameters
    # -------------------------------------------------------------------------
    SEED = 42

    # Gating & Sentinel Strategy
    GATING_THRESHOLD = 3.0  # Yards (Relaxed Quadratic Gating)
    SENTINEL_VALUE = -1.0  # For Ground distance

    # Label Smoothing
    LABEL_SMOOTHING_SIGMA = 1.0  # Gaussian smoothing sigma (timesteps)

    # Hard Negative Mining
    MINING_THRESHOLD = 0.05  # Probability threshold to consider a negative "Hard"

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000  # Number of plays to sample if DEBUG is True

    # -------------------------------------------------------------------------
    # 3. Feature Engineering Configuration
    # -------------------------------------------------------------------------
    # Collision-Aligned Vector Decomposition Features
    VECTOR_FEATURES = [
        "radial_velocity",
        "tangential_velocity",
        "radial_acceleration",
        "tangential_acceleration",
        "radial_accel_shock",  # Spectral Shock Feature
        "time_to_collision",
    ]

    # Basic Kinematics (Scalar)
    SCALAR_FEATURES = [
        "distance",
        "speed_p1",
        "speed_p2",
        "acceleration_p1",
        "acceleration_p2",
        "direction_p1",
        "direction_p2",
        "orientation_p1",
        "orientation_p2",
        "step",  # Temporal context
    ]

    # Final Feature List for Models
    FEATURES = SCALAR_FEATURES + VECTOR_FEATURES

    # -------------------------------------------------------------------------
    # 4. Model Hyperparameters (Tri-Ensemble)
    # -------------------------------------------------------------------------
    # Common Training Params
    NUM_BOOST_ROUND = 2000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # LightGBM (Leaf-wise)
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 256,  # High capacity
        "max_depth": 10,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "is_unbalance": True,  # Handle class imbalance internally
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
    }

    # XGBoost (Level-wise)
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 10,  # High capacity
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",  # Efficient for large data
        "device": "cuda",  # Use GPU if available
        "verbosity": 0,
        "seed": SEED,
        "n_jobs": -1,
        # scale_pos_weight will be calculated dynamically during training
    }

    @classmethod
    def get_feature_list(cls):
        """Returns the list of features to be used for training/inference."""
        return cls.FEATURES

    @classmethod
    def get_cache_path(cls, dataset_type):
        """Returns the cache path for a specific dataset type ('train', 'val', 'test')."""
        if dataset_type == "train":
            return cls.CACHE_TRAIN_FEATURES
        elif dataset_type == "val":
            return cls.CACHE_VAL_FEATURES
        elif dataset_type == "test":
            return cls.CACHE_TEST_FEATURES
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
