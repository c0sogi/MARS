import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_15"

# Ensure working directory exists for caching and outputs
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Tracking Data Paths
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")

# Output Paths
SUBMISSION_PATH = "submission.csv"

# =============================================================================
# GLOBAL HYPERPARAMETERS & CONSTANTS
# =============================================================================
RANDOM_STATE = 42

# Geometric Gating & Sentinel Strategy
GATING_THRESHOLD = 3.0  # Yards: Maximum distance to retain Player-Player pairs
GROUND_SENTINEL = -1.0  # Distance value assigned to Player-Ground interactions
CONTEXT_RADIUS = 2.0  # Yards: Radius to search for 3rd party neighbors

# Temporal Windowing
WINDOW_SIZE = 10  # Steps: Number of frames before/after (at 10Hz) to flatten

# Hard Negative Mining
HARD_NEGATIVE_THRESHOLD = (
    0.05  # Probability threshold to select hard negatives from Scout
)

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# LightGBM Parameters (High Capacity, Leaf-wise)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 256,
    "max_depth": 10,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "is_unbalance": True,  # Automatically handle class imbalance
    "verbosity": -1,
    "n_estimators": 2000,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

# XGBoost Parameters (High Capacity, Level-wise)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "max_depth": 10,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,  # Explicit weight for imbalance (approximate)
    "n_estimators": 2000,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "enable_categorical": False,
}

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Base Kinematics
BASE_FEATURES = [
    "distance",
    "speed_p1",
    "speed_p2",
    "acceleration_p1",
    "acceleration_p2",
    "direction_p1",
    "direction_p2",
    "orientation_p1",
    "orientation_p2",
    "x_position_p1",
    "y_position_p1",
    "x_position_p2",
    "y_position_p2",
]

# Derived Interaction Features
INTERACTION_FEATURES = ["speed_diff", "acc_diff", "direction_diff", "orientation_diff"]

# Invariant Extremum-Context Features (New Strategy)
CONTEXT_FEATURES = [
    "min_dist_3rd_party",  # Distance to closest 3rd party
    "max_closing_speed_3rd_party",  # Max closing speed of any neighbor
    "max_acceleration_3rd_party",  # Max acceleration of any neighbor
]

# Features to apply Temporal Windowing (Lag/Lead)
TEMPORAL_TARGET_FEATURES = [
    "distance",
    "speed_p1",
    "speed_p2",
    "acceleration_p1",
    "acceleration_p2",
    "min_dist_3rd_party",
]
