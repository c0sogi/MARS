import os

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_38"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
SEED = 42

# Training
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
EPOCHS = 20
PATIENCE = 3  # For Early Stopping

# Loss Function (Focal Loss)
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# Model Architecture & Regularization
NOISE_SIGMA = 0.05  # Standard deviation for Gaussian Input Noise
RESIDUAL_LAMBDA = 1.0  # Weight for Visual Stream residual connection
DROPOUT_RATE = 0.2
HIDDEN_DIMS = [512, 256, 128]  # Pyramidal backbone dimensions

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# Temporal Window: t-5 to t+5 (11 frames total)
WINDOW_SIZE = 5
TOTAL_FRAMES = 2 * WINDOW_SIZE + 1

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Raw columns to load from Player Tracking data
TRACKING_RAW_COLS = [
    "game_play",
    "game_key",
    "play_id",
    "nfl_player_id",
    "step",
    "datetime",
    "x_position",
    "y_position",
    "speed",
    "acceleration",
    "direction",
    "orientation",
    "sa",
]

# Raw columns to load from Baseline Helmets data
HELMET_RAW_COLS = [
    "game_play",
    "play_id",
    "nfl_player_id",
    "frame",
    "left",
    "top",
    "width",
    "height",
    "view",
]

# Base Kinematic Features (Per Timestamp)
# These will be expanded by the window size (e.g., x_position_t-5, ..., x_position_t+5)
KINEMATIC_BASE_FEATURES = [
    "x_position_1",
    "y_position_1",
    "speed_1",
    "acceleration_1",
    "direction_1",
    "orientation_1",
    "x_position_2",
    "y_position_2",
    "speed_2",
    "acceleration_2",
    "direction_2",
    "orientation_2",
    "distance",
    "closing_speed",
    "relative_angle",
]

# Base Visual Features (Per Timestamp)
# These will be expanded by the window size
VISUAL_BASE_FEATURES = [
    "left_1",
    "top_1",
    "width_1",
    "height_1",
    "area_1",
    "left_2",
    "top_2",
    "width_2",
    "height_2",
    "area_2",
]

# =============================================================================
# PHYSICAL CLAMPING RANGES
# =============================================================================
# Used for input clamping layer and feature engineering normalization
# Keys match substrings of feature names to apply broadly
CLAMPING_RANGES = {
    "x_position": (0.0, 120.0),
    "y_position": (0.0, 53.3),
    "speed": (0.0, 15.0),  # Yards/sec
    "acceleration": (0.0, 15.0),  # Yards/sec^2
    "distance": (0.0, 50.0),  # Yards
    "closing_speed": (-20.0, 20.0),  # Yards/sec
    "angle": (-180.0, 180.0),  # Degrees
    "orientation": (0.0, 360.0),  # Degrees
    "direction": (0.0, 360.0),  # Degrees
    "left": (0.0, 1280.0),  # Pixels
    "top": (0.0, 720.0),  # Pixels
    "width": (0.0, 1280.0),  # Pixels
    "height": (0.0, 720.0),  # Pixels
    "area": (0.0, 1280.0 * 720.0),  # Pixels^2
}

# =============================================================================
# COMPUTATIONAL CONFIG
# =============================================================================
NUM_WORKERS = 4
PIN_MEMORY = True
