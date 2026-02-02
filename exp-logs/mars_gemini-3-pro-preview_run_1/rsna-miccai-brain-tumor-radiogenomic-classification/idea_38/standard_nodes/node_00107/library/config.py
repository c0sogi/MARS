import os
import torch
import numpy as np
import random


# ==========================================
# Reproducibility
# ==========================================
def seed_everything(seed=42):
    """
    Sets the random seed for all relevant libraries to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set seed immediately upon import
SEED = 42
seed_everything(SEED)

# ==========================================
# File System Paths
# ==========================================
# Input Directories (Read-Only)
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata Paths (Pre-generated)
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Working Directory (Write Allowed)
# Specific directory for Idea 38 caching and checkpoints
WORKING_DIR = "./working/idea_38"
os.makedirs(WORKING_DIR, exist_ok=True)

# Cache Paths
CACHE_DIR = WORKING_DIR
TRAIN_CACHE_FILE = os.path.join(CACHE_DIR, "train_cache.npy")
VAL_CACHE_FILE = os.path.join(CACHE_DIR, "val_cache.npy")
TEST_CACHE_FILE = os.path.join(CACHE_DIR, "test_cache.npy")

# Model Checkpoints
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Submission
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# ==========================================
# Data Configuration
# ==========================================
IMG_SIZE = 224
NUM_CHANNELS = 3
# Selected modalities for the VAA Network (Channel 1, 2, 3)
# T1w is excluded as per strategy
SELECTED_MODALITIES = ["FLAIR", "T1wCE", "T2w"]

# ==========================================
# Model Hyperparameters
# ==========================================
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
NUM_CLASSES = 1
DROPOUT_RATE = 0.3  # Enforced dropout for regularization

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive weight decay
EARLY_STOPPING_PATIENCE = 5
NUM_FOLDS = 5

# ==========================================
# Compute Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# ==========================================
# Debugging / Development
# ==========================================
# Set DEBUG to True to run on a small subset of data for quick pipeline verification
DEBUG = False
DEBUG_DATA_SIZE = 50  # Number of samples to use in debug mode
