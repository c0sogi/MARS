import os
import torch

# ==========================================
# Directories and File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working directory for this specific idea/experiment
WORKING_DIR = "./working/idea_31"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# Metadata Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
IMG_SIZE = 32
IMG_CHANNELS = 3
NUM_CLASSES = 1  # Binary classification

# ==========================================
# Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-2

# Mixup Regularization
MIXUP_ALPHA = 0.2

# Deep Supervision
AUX_LOSS_WEIGHT = 0.4

# Stochastic Weight Averaging (SWA)
SWA_START_EPOCH = 20
SWA_LR = 5e-4

# Cross Validation
N_FOLDS = 5

# ==========================================
# Compute Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4


def setup_directories():
    """
    Creates necessary directories for cache, checkpoints, and submissions.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
