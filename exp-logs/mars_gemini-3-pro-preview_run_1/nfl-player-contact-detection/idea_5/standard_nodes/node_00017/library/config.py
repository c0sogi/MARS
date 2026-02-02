import os
import hashlib
import json


class Config:
    # --------------------------------------------------------------------------
    # Directories and Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Tracking Data Paths
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

    # Submission Path
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # --------------------------------------------------------------------------
    # Global Settings
    # --------------------------------------------------------------------------
    SEED = 42
    N_JOBS = 12  # Number of vCPUs available

    # --------------------------------------------------------------------------
    # Feature Engineering Parameters
    # --------------------------------------------------------------------------
    # Temporal Window: +/- these many steps (0.1s each)
    WINDOW_SIZE = 10

    # Topology: Distance threshold (yards) for defining a graph edge between players
    CONNECTION_RADIUS = 2.0

    # Topology: Whether to compute complex graph metrics (eigenvector centrality, etc.)
    USE_TOPOLOGY = False  # Disabled based on Lesson 15

    # Spatial Density: Count neighbors within radius (Cite solution_lesson_node_00012)
    USE_SPATIAL_DENSITY = True
    DENSITY_RADIUS = 1.5

    # --------------------------------------------------------------------------
    # Model Hyperparameters
    # --------------------------------------------------------------------------
    # LightGBM: Deep trees + Unbalance handling
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 256,  # High capacity
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "is_unbalance": True,  # Handle imbalance internally
        "verbosity": -1,
        "n_jobs": N_JOBS,
        "random_state": SEED,
    }

    # XGBoost: Deep trees + Scale Pos Weight
    # Note: scale_pos_weight estimate ~ ratio of negative/positive class
    # From analysis: 3352200 / 46228 ~= 72.5
    XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "max_depth": 10,  # High capacity
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 72.5,  # Handle imbalance
        "n_jobs": N_JOBS,
        "random_state": SEED,
        "enable_categorical": True,
        "tree_method": "hist",  # Efficient for large datasets
    }

    # Training Loop Settings
    N_ESTIMATORS = 2000
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------
    @staticmethod
    def get_feature_hash():
        """
        Generates a unique MD5 hash based on the current feature engineering configuration.
        This is used to create parameter-aware cache filenames.
        """
        config_dict = {
            "window_size": Config.WINDOW_SIZE,
            "connection_radius": Config.CONNECTION_RADIUS,
            "use_topology": Config.USE_TOPOLOGY,
            "use_spatial_density": Config.USE_SPATIAL_DENSITY,
            "density_radius": Config.DENSITY_RADIUS,
        }

        # Serialize to JSON with sorted keys to ensure consistency
        config_str = json.dumps(config_dict, sort_keys=True)

        # Create MD5 hash
        md5_hash = hashlib.md5(config_str.encode("utf-8")).hexdigest()

        return md5_hash
