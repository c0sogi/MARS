import os
import numpy as np


class Config:
    """
    Configuration for the Quadratic-Gated Vector-Spectral Anchored-Mining Ensemble (QGVSA-E).
    """

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_25"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Tracking Data
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_OUTPUT_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching (Parameter-Aware)
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.parquet")
    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.parquet")
    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.parquet")
    CACHE_HARD_NEGATIVES = os.path.join(WORKING_DIR, "hard_negative_indices.npy")
    CACHE_MODELS = os.path.join(WORKING_DIR, "models")

    os.makedirs(CACHE_MODELS, exist_ok=True)

    # -------------------------------------------------------------------------
    # Global Constants
    # -------------------------------------------------------------------------
    SEED = 42
    N_JOBS = 12

    # -------------------------------------------------------------------------
    # Physics & Gating Configuration
    # -------------------------------------------------------------------------
    # Quadratic Reachability Gating
    # Filter pairs where predicted min_dist < GATING_THRESHOLD
    GATING_THRESHOLD = 3.0  # Yards

    # Sentinel Value for Ground Interactions (Distance)
    GROUND_DISTANCE_SENTINEL = -1.0

    # Label Smoothing
    # Gaussian smoothing sigma for temporal labels (1 step = 0.1s)
    LABEL_SMOOTHING_SIGMA = 1.0

    # Spectral Feature Engineering
    # Window size for calculating RMS Energy of High-Passed Acceleration
    SPECTRAL_WINDOW_SIZE = 5

    # -------------------------------------------------------------------------
    # Feature Definition
    # -------------------------------------------------------------------------
    # Collision-Aligned Vector-Spectral Features
    FEATURES = [
        # 1. Primary Interaction Metric
        "distance",
        # 2. Collision-Aligned Decomposition (Vector)
        "radial_velocity",  # Impact Speed (Parallel to collision axis)
        "tangential_velocity",  # Shear Speed (Perpendicular to collision axis)
        "radial_acceleration",
        "tangential_acceleration",
        # 3. Spectral Features
        "radial_accel_spectral_energy",  # Transient Impact Shock
        # 4. Raw Relative Vectors (Geometry)
        "rel_v_x",
        "rel_v_y",
        # 5. Player State Features (P1 & P2)
        "speed_p1",
        "speed_p2",
        "acceleration_p1",
        "acceleration_p2",
        "direction_p1",
        "direction_p2",
        "orientation_p1",
        "orientation_p2",
        # 6. Interaction Context
        "speed_diff",
        "acc_diff",
    ]

    # -------------------------------------------------------------------------
    # Model Hyperparameters (Tri-Ensemble)
    # -------------------------------------------------------------------------
    # Common settings: Deep trees, high capacity

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
        "n_jobs": N_JOBS,
        "seed": SEED,
        "is_unbalance": True,  # Internal rebalancing
    }

    # 2. XGBoost (Level-wise growth)
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 10,
        "grow_policy": "depthwise",  # Level-wise
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "tree_method": "hist",
        # scale_pos_weight will be calculated dynamically during training based on ratio
    }

    # -------------------------------------------------------------------------
    # Training Curriculum
    # -------------------------------------------------------------------------
    NUM_BOOST_ROUND = 2000
    EARLY_STOPPING_ROUNDS = 50

    # Mining Settings
    # Threshold for a Scout to flag a sample as a "Hard Negative"
    HARD_NEGATIVE_THRESHOLD = 0.05

    # Ratio of Random Easy Negatives (Anchors) to include in Expert set
    # relative to the count of (Positives + Hard Negatives)
    ANCHOR_RATIO = 1.0
