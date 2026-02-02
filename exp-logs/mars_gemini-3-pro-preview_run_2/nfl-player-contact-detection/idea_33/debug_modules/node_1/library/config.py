import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_33"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # For data loading

# =============================================================================
# DATA PROCESSING
# =============================================================================
# Temporal Window: t-5 to t+5 (Total 11 frames)
WINDOW_PRE = 5
WINDOW_POST = 5
TOTAL_FRAMES = WINDOW_PRE + 1 + WINDOW_POST

# Sampling for debugging (set to None to use full data)
DEBUG_SAMPLE_SIZE = None

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
# Strict Invariance: No categorical entity embeddings (Team, Position, ID) are used.

# 1. Kinematic Features (Per Timestep)
# These are the columns expected in the kinematic input tensor (before flattening)
# Includes raw tracking data and derived physics terms.
KINEMATIC_FEATURES = [
    # Raw Tracking (Player 1 & 2)
    "x_position_1",
    "y_position_1",
    "speed_1",
    "acceleration_1",
    "orientation_1",
    "direction_1",
    "sa_1",
    "x_position_2",
    "y_position_2",
    "speed_2",
    "acceleration_2",
    "orientation_2",
    "direction_2",
    "sa_2",
    # Derived Physics
    "distance",  # Euclidean distance
    "distance_log1p",  # np.log1p(distance) for resolution
    "relative_speed",  # Closing speed
    "relative_angle",  # Shortest arc angle difference
    "is_ground",  # Binary flag for ground contact
]

# 2. Visual Features (Per Timestep)
# Derived from helmet bounding boxes via Max-Pooling strategy
VISUAL_FEATURES = [
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

# 3. Explicit Numerical Stability
# Ranges for clamping features to prevent outliers from destabilizing gradients.
# Applied during preprocessing and/or via a fixed Input Clamping Layer.
CLAMP_RANGES = {
    # Kinematic Clamps
    "speed": (0.0, 40.0),  # Yards/sec (Safe upper bound)
    "acceleration": (0.0, 50.0),  # Yards/sec^2
    "sa": (-50.0, 50.0),  # Signed acceleration
    "relative_speed": (-50.0, 50.0),  # Closing speed
    "distance": (0.0, 120.0),  # Max field dimension
    "distance_log1p": (0.0, 5.0),  # log(120) approx 4.8
    "relative_angle": (0.0, 180.0),  # Shortest arc (0-180)
    "orientation": (0.0, 360.0),
    "direction": (0.0, 360.0),
    # Visual Clamps (Pixel coordinates/dimensions)
    "left": (0, 1280),
    "top": (0, 720),
    "width": (0, 1280),
    "height": (0, 720),
    "area": (0, 1280 * 720),
}

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Architecture
HIDDEN_DIM_KIN = 256
HIDDEN_DIM_VIS = 64
DROPOUT_RATE = 0.3

# Training
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3

# Loss Function: Focal Loss
# alpha=0.25 balances easy negatives
# gamma=2.0 focuses on hard examples
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# Residual Fusion
# Weight for the visual stream logit in the final sum
VISUAL_LOSS_WEIGHT = 1.0

# =============================================================================
# INFERENCE & EVALUATION
# =============================================================================
# Grid search parameters for threshold optimization on validation set
THRESHOLD_START = 0.01
THRESHOLD_END = 0.99
THRESHOLD_STEPS = 100
