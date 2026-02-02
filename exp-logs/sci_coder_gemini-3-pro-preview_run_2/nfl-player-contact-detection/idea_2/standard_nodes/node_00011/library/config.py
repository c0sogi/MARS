import os

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Files
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache Files (Parquet/NPY)
CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features_seq.parquet")
CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features_seq.parquet")
CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features_seq.parquet")
CACHE_SCALER = os.path.join(
    WORKING_DIR, "scaler.joblib"
)  # Using joblib for sklearn scaler
MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "dstcn_model.pth")

# =============================================================================
# DATA PIPELINE CONFIGURATION
# =============================================================================
# Temporal Window: t-5 to t+5 (11 frames total at 10Hz)
WINDOW_SIZE = 11
HALF_WINDOW = (WINDOW_SIZE - 1) // 2

# Columns to read from tracking data
TRACKING_COLS = [
    "game_play",
    "step",
    "nfl_player_id",
    "x_position",
    "y_position",
    "speed",
    "acceleration",
    "orientation",
    "direction",
    "sa",  # Signed acceleration
]

# Features to be used by the model
# These must be generated during feature engineering
FEATURE_COLS = [
    "log_distance",  # Log1p of Euclidean distance
    "clamped_speed",  # Closing speed with denominator clamping
    "acceleration_1",  # Player 1 acceleration
    "acceleration_2",  # Player 2 acceleration (0 if ground)
    "jerk_1",  # Derivative of acceleration P1
    "jerk_2",  # Derivative of acceleration P2
    "orientation_diff",  # Difference in orientation
    "direction_diff",  # Difference in direction of motion
    "is_ground",  # Binary flag for ground contact
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 15
PATIENCE = 3  # For Early Stopping

# Architecture
INPUT_CHANNELS = len(FEATURE_COLS)
HIDDEN_CHANNELS = 64
KERNEL_SIZE = 3
DROPOUT_RATE = 0.2

# Loss Weighting
# Based on ~1:72 imbalance ratio
POS_WEIGHT = 72.0

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
# Default threshold (will be optimized based on validation set)
DEFAULT_THRESHOLD = 0.5
