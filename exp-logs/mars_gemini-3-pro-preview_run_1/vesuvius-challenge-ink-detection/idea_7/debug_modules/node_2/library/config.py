import os
import torch
import numpy as np

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_7"

# Ensure working directory structure exists
os.makedirs(WORKING_DIR, exist_ok=True)

CACHE_DIR = os.path.join(WORKING_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

PREDICTIONS_DIR = os.path.join(WORKING_DIR, "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Submission Path (Root directory as per competition format)
SUBMISSION_PATH = "./submission.csv"

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Data Dimensions
PATCH_SIZE = 512
Z_DIM = 65  # Number of slices in the z-direction

# Training Configuration
# Strict requirement: Batch Size 32 to maximize iterations/updates
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 4  # Optimized for 12 vCPUs
SEED = 42

# Model Configuration
# Base channels reduced to 32 to accommodate Batch Size 32 on GPU
MODEL_BASE_CHANNELS = 32
POS_WEIGHT = 2.0  # Weight for positive class in BCE Loss

# Compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# UTILITIES
# =============================================================================


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
