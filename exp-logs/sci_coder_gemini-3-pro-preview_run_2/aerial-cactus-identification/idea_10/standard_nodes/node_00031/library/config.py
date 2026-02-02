import os
import random
import numpy as np
import torch

# =============================================================================
# DIRECTORY & FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Specific working directory for this strategy
IDEA_NAME = "idea_10"
IDEA_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
OUTPUT_DIR = os.path.join(IDEA_DIR, "submission")
MODEL_DIR = IDEA_DIR

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
IMAGE_SIZE = (32, 32)
NUM_CLASSES = 1
NUM_WORKERS = 4

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Custom Narrow SE-Multi-Scale ResNet settings
CHANNEL_CONFIG = [16, 32, 64]  # Narrow width for efficiency
USE_SE_BLOCK = True  # Squeeze-and-Excitation for attention
USE_MULTI_SCALE = True  # Aggregating features from Stage 2 and 3
DROPOUT_RATE = 0.0

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEEDS = [0, 1, 2, 3, 4]  # Homogeneous Seed Averaging
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def setup_directories():
    """
    Creates the necessary working directories for the current idea.
    """
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # print(f"Directories initialized at: {IDEA_DIR}")


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic operations for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


# Automatically setup directories when config is imported
setup_directories()
