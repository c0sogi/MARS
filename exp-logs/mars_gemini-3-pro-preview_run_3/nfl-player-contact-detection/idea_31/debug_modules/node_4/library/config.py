import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the contact detection pipeline.
    Serves as the single source of truth for cache invalidation and experiment settings.
    """

    # =========================================================================
    # PATHS
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_31"

    # Input Files
    TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
    TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Generated previously)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # GLOBAL SETTINGS
    # =========================================================================
    SEED = 42
    USE_GPU = torch.cuda.is_available()
    NUM_WORKERS = 4

    # Debugging / Development
    # Set to a smaller integer (e.g., 5000) to limit dataset size for rapid iteration.
    # Set to None for full training.
    DEBUG_SAMPLE_SIZE = None

    # =========================================================================
    # FEATURE ENGINEERING CONFIGURATION
    # =========================================================================
    # Time window settings for flattening and lags
    # Exponential lags for sparse temporal sampling (e.g., t +/- 1, 2, 4, 8, 15)
    EXP_LAGS = [1, 2, 4, 8, 15]

    # Window size for flattening trajectories (e.g., +/- 10 frames around contact)
    FLATTEN_WINDOW = 10

    # Stream A: Interaction Model (Player-Player)
    # Features: Relational Scalars, System Energy, Visual Consensus
    STREAM_A_FEATURES = {
        "relational": ["distance", "closure_rate"],
        "energy": ["speed", "acceleration", "sa"],  # Applied to p1 and p2
        "visual": [
            "sideline_iou",
            "endzone_iou",
            "max_iou",
            "min_iou",
            "iou_diff",
            "visual_looming_rate",
        ],
        "meta": ["step"],
    }

    # Stream B: Impact Model (Player-Ground)
    # Features: Hybrid Context (Field-Centric + Ego-Dynamics)
    STREAM_B_FEATURES = {
        "field_centric": [
            "x_position",
            "y_position",
            "speed",
            "acceleration",
            "direction",
            "orientation",
        ],
        "ego_dynamics": ["v_surge", "v_sway", "a_surge", "a_sway", "j_surge", "j_sway"],
        "meta": ["step"],
    }

    # =========================================================================
    # MODEL HYPERPARAMETERS (XGBoost)
    # =========================================================================
    # Common settings
    COMMON_XGB_PARAMS = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "gpu_hist" if USE_GPU else "hist",
        "random_state": SEED,
        "n_jobs": -1,
    }

    # Stream A: Interaction Model
    # Standard depth to capture complex cross-modal interactions
    STREAM_A_PARAMS = {
        **COMMON_XGB_PARAMS,
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 2000,  # with early stopping
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,  # Handled via sampling, but can be tuned
    }

    # Stream B: Impact Model
    # Shallower depth to prevent overfitting on sensor noise, relying on robust engineered features
    STREAM_B_PARAMS = {
        **COMMON_XGB_PARAMS,
        "max_depth": 7,  # 6-8 range
        "learning_rate": 0.05,
        "n_estimators": 2000,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,  # Slightly higher regularization
        "reg_lambda": 1.5,
    }

    # Training settings
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100

    # Sampling Strategy
    # Ratio of Negative to Positive samples (10:1)
    NEG_POS_RATIO = 10.0

    # =========================================================================
    # METHODS
    # =========================================================================
    @classmethod
    def setup(cls):
        """
        Initialize the environment: create directories and set seeds.
        """
        # Create working directory
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

        # Create submission directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)

        # Set reproducible seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)

    @classmethod
    def get_config_hash(cls):
        """
        Returns a dictionary representation of the configuration for hashing.
        Used to invalidate caches if parameters change.
        """
        return {
            "lags": cls.EXP_LAGS,
            "window": cls.FLATTEN_WINDOW,
            "features_a": cls.STREAM_A_FEATURES,
            "features_b": cls.STREAM_B_FEATURES,
            "params_a": cls.STREAM_A_PARAMS,
            "params_b": cls.STREAM_B_PARAMS,
            "seed": cls.SEED,
        }


# Initialize environment immediately upon import
Config.setup()
