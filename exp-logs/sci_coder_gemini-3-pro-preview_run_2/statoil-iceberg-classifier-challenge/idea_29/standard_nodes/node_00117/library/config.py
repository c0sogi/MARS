import os
import random
import numpy as np
import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for the current idea (WB-DIN)
WORKING_DIR = "./working/idea_29"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
IMG_SIZE = 75
# Input channels: Band 1 (HH), Band 2 (HV), Mean((B1+B2)/2)
N_CHANNELS = 3
# Target classes: Ship (0) vs Iceberg (1)
N_CLASSES = 1

# =============================================================================
# MODEL ARCHITECTURE (WB-DIN)
# =============================================================================
# Contracting Width Profile for the 4 stages [64, 128, 128, 32]
# Final output channels = 32 * 2 (DualPool) = 64
WIDTH_PROFILE = [64, 128, 128, 32]
# High dropout rate to regularize the backbone
DROPOUT_RATE = 0.5
# Enable Metadata Branch for incidence angle
USE_METADATA = True
METADATA_DIM = 1

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
N_FOLDS = 5
SEED = 42
BATCH_SIZE = 32
# "Low and Slow" initialization
LEARNING_RATE = 2e-4
# ReduceLROnPlateau settings
LR_FACTOR = 0.5
LR_PATIENCE = 5
MIN_LR = 1e-6
# Early Stopping settings
EPOCHS = 100
PATIENCE = 15

# =============================================================================
# AUGMENTATION CONFIGURATION
# =============================================================================
# Rotational invariance angles
ROTATION_ANGLES = [0, 90, 180, 270]
DO_HORIZONTAL_FLIP = True
DO_VERTICAL_FLIP = False  # Excluded based on constraints

# =============================================================================
# DEBUGGING & DEVELOPMENT
# =============================================================================
# Set DEBUG to True to run on a small subset of data
DEBUG = False
DEBUG_SIZE = 100


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def setup_directories():
    """Creates necessary directories for the project."""
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# Automatically setup directories when config is imported
setup_directories()
