import os


class PathConfig:
    """
    Defines file paths for inputs, metadata, cached artifacts, and models.
    Ensures working and submission directories exist.
    """

    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Raw Input Files
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")

    # Generated Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cached Feature Files (Parquet/Numpy)
    TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.parquet")
    VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.parquet")
    TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.parquet")
    HARD_NEGATIVE_INDICES_PATH = os.path.join(WORKING_DIR, "hard_negative_indices.npy")

    # Model Artifacts
    SCOUT_LGBM_PATH = os.path.join(WORKING_DIR, "scout_lgbm.joblib")
    SCOUT_XGB_PATH = os.path.join(WORKING_DIR, "scout_xgb.joblib")
    EXPERT_LGBM_PATH = os.path.join(WORKING_DIR, "expert_lgbm.joblib")
    EXPERT_XGB_PATH = os.path.join(WORKING_DIR, "expert_xgb.joblib")

    # Final Submission
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")


class FeatureConfig:
    """
    Configuration for Physics-Enhanced Feature Engineering and Geometric Gating.
    """

    SEED = 42

    # Gating and Windowing
    WINDOW_SIZE = 10  # +/- 10 steps (1.0 second context)
    GATING_DISTANCE = 3.0  # Yards. Pairs further than this are discarded (Stage 0).
    SENTINEL_VALUE = -1.0  # Value for physics features when undefined (e.g., diverging)

    # Tracking columns to use for feature generation
    TRACKING_COLS = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Specific Physics & Interaction Features
    # These are computed for the primary pair only
    PHYSICS_FEATURES = [
        "time_to_collision",  # Distance / Closing Speed
        "kinetic_energy_proxy",  # Relative Speed ^ 2
        "jerk",  # Derivative of acceleration
        "angular_jerk",  # Derivative of orientation change
    ]

    # Contextual Features
    CONTEXT_FEATURES = ["spatial_density"]  # Count of other players nearby

    # Ground Contact Flag
    GROUND_FEATURE = "is_ground"


class ModelConfig:
    """
    Hyperparameters for the Dual-Stage Heterogeneous Ensemble.
    """

    SEED = 42

    # Mining Strategy
    HARD_NEGATIVE_THRESHOLD = 0.05  # Probability threshold for mining hard negatives

    # -------------------------------------------------------------------------
    # STAGE 1: SCOUT TIER (Balanced Data, Standard Capacity)
    # -------------------------------------------------------------------------
    SCOUT_LGBM_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "n_estimators": 500,
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    SCOUT_XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_estimators": 500,
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # -------------------------------------------------------------------------
    # STAGE 2: EXPERT TIER (Imbalanced Data + Hard Negatives, High Capacity)
    # -------------------------------------------------------------------------
    EXPERT_LGBM_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,  # Slower learning for fine-tuning
        "num_leaves": 256,  # High capacity for complex boundaries
        "max_depth": 10,  # Deep trees
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "n_estimators": 2000,
        "is_unbalance": True,  # Explicit imbalance handling
        "verbose": -1,
        "random_state": SEED,
        "n_jobs": -1,
    }

    EXPERT_XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.02,
        "max_depth": 10,  # Deep trees
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "n_estimators": 2000,
        "scale_pos_weight": 10,  # Heuristic for imbalance (approx 1:70 raw, 1:10 after mining)
        "tree_method": "hist",
        "random_state": SEED,
        "n_jobs": -1,
    }
