import os
import torch

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_27"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Data Files
TRAIN_TRACKING_PATH = os.path.join(INPUT_DIR, "train_player_tracking.csv")
TEST_TRACKING_PATH = os.path.join(INPUT_DIR, "test_player_tracking.csv")
TRAIN_HELMETS_PATH = os.path.join(INPUT_DIR, "train_baseline_helmets.csv")
TEST_HELMETS_PATH = os.path.join(INPUT_DIR, "test_baseline_helmets.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Generated Pre-split)
METADATA_TRAIN = os.path.join(METADATA_DIR, "train.csv")
METADATA_VAL = os.path.join(METADATA_DIR, "validation.csv")
METADATA_TEST = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# DATA PROCESSING CONFIGURATION
# =============================================================================

# Random Seed for Reproducibility
SEED = 42

# Windowing Logic
# Window size 5 means: t-5, t-4, ..., t, ..., t+4, t+5 (Total 11 frames)
WINDOW_SIZE = 5
TOTAL_WINDOW_LEN = 2 * WINDOW_SIZE + 1

# Numerical Stability Constraints
# Explicit clamping range for kinematic features to prevent gradient explosions
CLAMP_MIN = -50.0
CLAMP_MAX = 50.0

# Feature Columns
# Categorical features for Entity Embeddings
CAT_COLS = ["position", "team"]

# Continuous Kinematic Features (to be windowed)
# These will be clamped and flattened: len(KINEMATIC_COLS) * TOTAL_WINDOW_LEN
KINEMATIC_COLS = [
    "x_position",
    "y_position",
    "speed",
    "acceleration",
    "direction",
    "orientation",
    "sa",
]

# Visual Features (Visual Stream)
# These are processed via Max-Pooling per timestamp
VISUAL_COLS = ["left", "width", "top", "height"]

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================

# Embedding Dimensions
EMBEDDING_DIM = 8

# Kinematic Stream (Deep Residual MLP)
HIDDEN_DIM_KIN = 256
NUM_LAYERS_KIN = 3

# Visual Stream (Shallow MLP)
HIDDEN_DIM_VIS = 64
NUM_LAYERS_VIS = 2

# General Model Settings
DROPOUT_RATE = 0.1

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# Compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Adjust based on vCPU availability (12 vCPUs available)

# Hyperparameters
BATCH_SIZE = 2048
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 15
EARLY_STOPPING_PATIENCE = 3

# Loss Function: Focal Loss
# alpha=0.25 balances positive/negative class importance
# gamma=2.0 focuses learning on hard examples
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_training_settings(debug=False):
    """
    Returns a dictionary of training settings, allowing for debug overrides.
    """
    settings = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "learning_rate": LEARNING_RATE,
    }

    if debug:
        settings["epochs"] = 2
        settings["batch_size"] = 1024
        # Reduce workers for debugging to avoid overhead
        settings["num_workers"] = 0

    return settings
