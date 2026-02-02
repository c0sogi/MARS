import os

# ==============================================================================
# Paths & Directories
# ==============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_28"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Dataset Paths
TRAIN_LABELS_PATH = os.path.join(INPUT_DIR, "train_labels.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Tracking Data
TRACKING_PATHS = {
    "train": os.path.join(INPUT_DIR, "train_player_tracking.csv"),
    "test": os.path.join(INPUT_DIR, "test_player_tracking.csv"),
}

# Helmet Data
HELMET_PATHS = {
    "train": os.path.join(INPUT_DIR, "train_baseline_helmets.csv"),
    "test": os.path.join(INPUT_DIR, "test_baseline_helmets.csv"),
}

# Metadata (Generated)
METADATA_PATHS = {
    "train": os.path.join(METADATA_DIR, "train.csv"),
    "validation": os.path.join(METADATA_DIR, "validation.csv"),
    "test": os.path.join(METADATA_DIR, "test.csv"),
}

# Video Metadata
VIDEO_METADATA_PATHS = {
    "train": os.path.join(INPUT_DIR, "train_video_metadata.csv"),
    "test": os.path.join(INPUT_DIR, "test_video_metadata.csv"),
}

# ==============================================================================
# Global Configuration
# ==============================================================================
SEED = 42
NEG_POS_RATIO = 10.0  # Targeted Majority Undersampling Ratio (10:1)
EARLY_STOPPING_ROUNDS = 50

# Temporal Pyramids
# Flatten features at sparse lags: t, t+/-1, t+/-2, t+/-4, t+/-8, t+/-15
LAG_OFFSETS = [-15, -8, -4, -2, -1, 0, 1, 2, 4, 8, 15]

# ==============================================================================
# Feature Definitions
# ==============================================================================

# STREAM A: Interaction (Player-Player)
# Logic: Relational Geometry + System Energy + Vision
STREAM_A_FEATURES = [
    # Relational Primitives
    "distance",
    "closure_rate",  # -Delta d / Delta t
    # System Energy (Absolute)
    "speed_p1",
    "speed_p2",
    "acceleration_p1",
    "acceleration_p2",
    # Visual Consensus
    "iou_sideline",
    "iou_endzone",
    "iou_max",
    "iou_min",
    "iou_diff",
]

# STREAM B: Impact (Player-Ground)
# Logic: Hybrid Context (Field-Centric) + Ego-Centric Physics
# Explicitly excludes visual features
STREAM_B_FEATURES = [
    # Field-Centric Context (Absolute)
    "x_position",
    "y_position",
    "speed",
    "acceleration",
    "direction",
    "orientation",
    # Ego-Centric Physics (Derived)
    "v_surge",  # Velocity projected on orientation
    "v_sway",  # Velocity projected orthogonal to orientation
    "a_surge",  # Derivative of v_surge
    "a_sway",  # Derivative of v_sway
    "j_surge",  # Derivative of a_surge (Jerk)
    "j_sway",  # Derivative of a_sway (Jerk)
]

# Raw Columns required from Tracking Data
REQUIRED_TRACKING_COLS = [
    "game_play",
    "step",
    "nfl_player_id",
    "x_position",
    "y_position",
    "speed",
    "acceleration",
    "direction",
    "orientation",
    "sa",
]

# ==============================================================================
# Model Hyperparameters (XGBoost)
# ==============================================================================

# Stream A: Interaction
# Max Depth 6 for complex interactions
XGB_PARAMS_STREAM_A = {
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": SEED,
    "n_jobs": -1,
}

# Stream B: Impact
# Max Depth 8 for robust physics signatures
XGB_PARAMS_STREAM_B = {
    "n_estimators": 5000,
    "learning_rate": 0.02,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": SEED,
    "n_jobs": -1,
}

# ==============================================================================
# Validation Mapping
# ==============================================================================
# Explicitly map validation mode to train tracking data
TRACKING_FILE_MAP = {
    "train": TRACKING_PATHS["train"],
    "validation": TRACKING_PATHS["train"],
    "test": TRACKING_PATHS["test"],
}

HELMET_FILE_MAP = {
    "train": HELMET_PATHS["train"],
    "validation": HELMET_PATHS["train"],
    "test": HELMET_PATHS["test"],
}
