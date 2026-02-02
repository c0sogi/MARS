import os
import torch
import random
import numpy as np

# =============================================================================
# GLOBAL PATHS
# =============================================================================

# Input Directories (Read-Only)
INPUT_DIR = "./input"
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

# Metadata Files (Pre-generated)
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Working Directory (Write Access)
WORKING_DIR = "./working"
IDEA_ID = "idea_32"
BASE_OUTPUT_DIR = os.path.join(WORKING_DIR, IDEA_ID)

# Sub-directories for specific artifacts
CACHE_DIR = os.path.join(BASE_OUTPUT_DIR, "cache")
CHECKPOINT_DIR = os.path.join(BASE_OUTPUT_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(BASE_OUTPUT_DIR, "submission")

# Final Submission File
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

# Data
IMAGE_SIZE = 32
NUM_CHANNELS = 3
NUM_CLASSES = 1  # Binary classification

# Training
SEED = 42
BATCH_SIZE = 128  # A100 can handle large batches, 32x32 images are small
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Cross-Validation
NUM_FOLDS = 5

# Regularization
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.0  # Using Mixup instead

# SWA (Stochastic Weight Averaging)
SWA_START_EPOCH = 20
SWA_LR = 1e-4

# Architecture List
# These keys correspond to the model classes to be implemented
MODEL_ARCHITECTURES = ["RepVGG", "ResNet", "NeXt"]

# Compute
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def setup_directories():
    """
    Creates necessary directories for cache, checkpoints, and submissions.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    print(f"Directories setup at {BASE_OUTPUT_DIR}")


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")
