import os
import torch
import numpy as np
import random

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA PIPELINE CONFIGURATION
# =============================================================================
# Window size for temporal context (t-5 to t+5)
WINDOW_SIZE = 11
HALF_WINDOW = WINDOW_SIZE // 2

# Columns to load from raw tracking data
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

# The final list of features used as input to the model per timestep
# These correspond to the "Explicit Relative Kinematics" and physical invariants
INPUT_FEATURES = [
    "distance",  # Log-transformed Euclidean distance
    "speed_1",  # Speed of player 1
    "speed_2",  # Speed of player 2 (or 0 for ground)
    "accel_1",  # Acceleration of player 1
    "accel_2",  # Acceleration of player 2 (or 0 for ground)
    "rel_speed",  # Magnitude of relative velocity vector
    "rel_accel",  # Magnitude of relative acceleration vector
    "closing_speed",  # Project of relative velocity onto distance vector
    "is_ground",  # Binary flag: 1 if Player 2 is Ground, else 0
]

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Architecture dimensions
HIDDEN_DIM = 256  # Dimension of the feature embedding and attention layers
NUM_HEADS = 4  # Number of attention heads
DROPOUT = 0.1  # Dropout rate for regularization
FF_DIM = 512  # Feed-forward dimension in attention blocks (if used)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
SEED = 42
BATCH_SIZE = 2048  # Large batch size for efficient processing of tabular sequences
LEARNING_RATE = 1e-3  # Standard AdamW learning rate
EPOCHS = 15  # Maximum number of training epochs
PATIENCE = 3  # Early stopping patience

# Focal Loss Parameters (to handle class imbalance)
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_device():
    """Returns the appropriate device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed=SEED):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize the system state immediately upon import
seed_everything()
