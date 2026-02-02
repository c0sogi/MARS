import os


class Config:
    """
    Centralized configuration for the Variable-Resolution Contextual Mining Ensemble (VRC-ME).
    Handles file paths, hyperparameters, and dynamic feature list generation.
    """

    # =========================================================================
    # 1. Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_8"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Paths
    TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
    TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # 2. Global Constants
    # =========================================================================
    SEED = 42
    WINDOW_SIZE = 10  # +/- 10 steps (1.0 second context)
    MINING_THRESHOLD = 0.01  # Probability threshold for hard-negative mining

    # Debugging / Development switches
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50000  # Number of rows to use if DEBUG is True

    # =========================================================================
    # 3. Feature Definitions
    # =========================================================================

    # Base columns from tracking data required for engineering
    TRACKING_COLS = [
        "x_position",
        "y_position",
        "speed",
        "direction",
        "orientation",
        "acceleration",
        "sa",
    ]

    # Kinematic features: Computed relative between players (or player-ground)
    # These are always windowed in both Tier 1 and Tier 2
    KINEMATIC_FEATURES_BASE = [
        "distance",
        "speed_p1",
        "speed_p2",
        "speed_diff",
        "acc_p1",
        "acc_p2",
        "acc_diff",
        "orient_p1",
        "orient_p2",
        "orient_diff",
        "dir_p1",
        "dir_p2",
        "dir_diff",
    ]

    # Physics derivatives: Computed from windowed data
    # These are effectively windowed features
    PHYSICS_FEATURES_BASE = ["jerk_p1", "jerk_p2", "angular_jerk_p1", "angular_jerk_p2"]

    # Context features: Spatial density and flow
    # Tier 1: Computed at t=0 only
    # Tier 2: Computed for +/- window
    CONTEXT_FEATURES_BASE = [
        "spatial_density",  # Number of players within interaction radius
        "cluster_speed",  # Average speed of local cluster
    ]

    @classmethod
    def get_feature_list(cls, tier=1):
        """
        Generates the exhaustive list of feature column names based on the Tier.

        Args:
            tier (int): 1 for Scout (Low-Res Context), 2 for Expert (High-Res Context).

        Returns:
            list: List of string column names.
        """
        features = []

        # --- 1. Kinematics & Physics (High-Res in both Tiers) ---
        # We include t=0 and Lag/Lead steps
        base_kinematics = cls.KINEMATIC_FEATURES_BASE + cls.PHYSICS_FEATURES_BASE

        # Add t=0
        features.extend(base_kinematics)

        # Add Lags and Leads
        for step in range(1, cls.WINDOW_SIZE + 1):
            for feat in base_kinematics:
                features.append(f"{feat}_lag{step}")
                features.append(f"{feat}_lead{step}")

        # --- 2. Context (Variable Resolution) ---
        # Add t=0 (Always present)
        features.extend(cls.CONTEXT_FEATURES_BASE)

        if tier == 2:
            # Add Lags and Leads ONLY for Tier 2
            for step in range(1, cls.WINDOW_SIZE + 1):
                for feat in cls.CONTEXT_FEATURES_BASE:
                    features.append(f"{feat}_lag{step}")
                    features.append(f"{feat}_lead{step}")

        return features

    # Pre-computed feature lists for easy access
    TIER1_FEATURES = []  # Populated below
    TIER2_FEATURES = []  # Populated below

    # =========================================================================
    # 4. Model Hyperparameters
    # =========================================================================

    # Stage 1: Scout (LightGBM)
    # Objective: Maximize Recall / AUC to find hard negatives.
    # Lightweight to handle full dataset.
    SCOUT_LGBM_PARAMS = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "n_estimators": 1000,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Stage 2: Expert (LightGBM)
    # Objective: Maximize Precision/MCC on difficult subset.
    # Higher capacity (more leaves).
    EXPERT_LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.02,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "is_unbalance": True,  # Critical for handling the mined imbalance
        "n_estimators": 2000,
        "random_state": SEED,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Stage 2: Expert (XGBoost)
    # Heterogeneous ensemble partner.
    EXPERT_XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.02,
        "max_depth": 8,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "scale_pos_weight": 5,  # Moderate balancing
        "n_estimators": 2000,
        "random_state": SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }

    # Training Loop Settings
    EARLY_STOPPING_ROUNDS = 50
    VERBOSE_EVAL = 100


# Populate the feature lists
Config.TIER1_FEATURES = Config.get_feature_list(tier=1)
Config.TIER2_FEATURES = Config.get_feature_list(tier=2)
