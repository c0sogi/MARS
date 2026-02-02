import os
import torch
import random
import numpy as np

# =============================================================================
# Global Configuration for HD-RDN Denoising Task
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Paths and Directories
# -----------------------------------------------------------------------------
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
SUBMISSION_DIR = "./submission"

# Ensure necessary writeable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output File Paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "rdn_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths (for deterministic data processing)
# We use .npy format for efficient storage of high-density patches
TRAIN_PATCHES_CACHE = os.path.join(WORKING_DIR, "train_patches.npy")
TRAIN_TARGETS_CACHE = os.path.join(WORKING_DIR, "train_targets.npy")
VAL_PATCHES_CACHE = os.path.join(WORKING_DIR, "val_patches.npy")
VAL_TARGETS_CACHE = os.path.join(WORKING_DIR, "val_targets.npy")

# -----------------------------------------------------------------------------
# 2. Data Parameters
# -----------------------------------------------------------------------------
# Image Specifications
IMG_CHANNELS = 1  # Grayscale
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0

# High-Density Patch Extraction Strategy
# Stride is significantly smaller than patch size to generate overlapping patches
PATCH_SIZE = 50
STRIDE = 10

# -----------------------------------------------------------------------------
# 3. Model Parameters (Residual Dense Network - RDN)
# -----------------------------------------------------------------------------
# Architecture hyperparameters for the RDN backbone
RDN_GROWTH_RATE = 64  # Growth rate (k)
RDN_NUM_FEATURES = 64  # Initial number of features (G0)
RDN_NUM_BLOCKS = 16  # Number of Residual Dense Blocks (D)
RDN_LAYERS_PER_BLOCK = 8  # Number of dense layers per block (C)
RDN_KERNEL_SIZE = 3  # Kernel size for convolutions

# -----------------------------------------------------------------------------
# 4. Training Parameters
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0

# Optimization & Scheduling
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3

# DataLoader Settings
NUM_WORKERS = 4
PIN_MEMORY = True


# -----------------------------------------------------------------------------
# 5. Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Apply seed immediately on import
set_seed()
