import os
import torch
import numpy as np
import random

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
SUBMISSION_FILE = "./submission.csv"

# Metadata Paths
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Data Configuration
# ==========================================
Z_DIM = 65  # Depth of the volume (number of slices)
PATCH_SIZE = 256  # Spatial dimension (H, W) for model input
INFERENCE_STRIDE = 128  # Stride for sliding window inference
NUM_WORKERS = 2  # Number of subprocesses for data loading

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
POS_WEIGHT = 2.0  # BCE Loss weight for the positive (ink) class

# ==========================================
# System Configuration
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def setup_reproducibility(seed=SEED):
    """
    Sets fixed random seeds for python, numpy, and torch to ensure reproducible results.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_directories():
    """
    Creates the necessary cache directory for the current idea.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
