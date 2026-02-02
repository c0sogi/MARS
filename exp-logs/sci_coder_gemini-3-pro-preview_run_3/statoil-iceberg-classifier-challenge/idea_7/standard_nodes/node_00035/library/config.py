import os
import torch
import numpy as np
import random

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/optimization"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# File paths derived from metadata
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw data paths
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Output paths
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
IMAGE_SIZE = 75
NUM_CHANNELS = 3  # HH, HV, Average
NUM_CLASSES = 1  # Binary classification (sigmoid output)

# Debugging flags
DEBUG = False
DEBUG_SUBSET_SIZE = 100  # Number of samples to use if DEBUG is True

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================
# As per Hierarchical Max-Pooling CNN design
CNN_CHANNELS = [64, 64, 128, 128]  # Channel widths for the 4 blocks
DENSE_UNITS = 512
DROPOUT_RATE = 0.2  # Applied after dense activation

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3  # Constant learning rate
PATIENCE = 10  # Early stopping patience
SEED = 42

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2  # For DataLoader


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
