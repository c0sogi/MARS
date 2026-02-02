import os
import torch
import random
import numpy as np

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
# Input Directories
INPUT_DIR = "./input"
TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

# Metadata Directories (Pre-generated)
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working & Output Directories
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_25")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary writable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Image Dimensions
IMAGE_SIZE = (768, 768)  # Height, Width

# Input Channels: 3
# Channel 0: Mammogram Image (Grayscale)
# Channel 1: Age Map (Spatially broadcasted metadata)
# Channel 2: Implant Map (Spatially broadcasted metadata)
IN_CHANNELS = 3

# Dataloader Settings
BATCH_SIZE = 6  # Adjusted for 768x768 Siamese Network on A100-40GB
NUM_WORKERS = 12
PIN_MEMORY = True

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Backbone
BACKBONE = "tf_efficientnet_b2"
PRETRAINED = True
DROP_RATE = 0.3
DROP_PATH_RATE = 0.2

# Architecture Specifics
USE_FPN = True
USE_ASYMMETRY_GATING = True
FPN_CHANNELS = 128

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# General
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 10
DEBUG = False  # Set to True to run on a small subset for testing

# Optimization
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Standard for AdamW
OPTIMIZER = "AdamW"
SCHEDULER = "CosineAnnealingLR"
MIN_LR = 1e-6

# Loss Function Settings
# Aggressive positive weighting to handle 1:47 imbalance
POS_WEIGHT = 47.0

# Gradient Strategy
# Explicitly disabled to allow large updates for the minority class
GRADIENT_CLIPPING = False


# =============================================================================
# SYSTEM UTILITIES
# =============================================================================
def set_seed(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to: {seed}")
