import os
import torch
import numpy as np
import random

# =============================================================================
# Directories and Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
PROJECT_NAME = "idea_4"
PROJECT_DIR = os.path.join(WORKING_DIR, PROJECT_NAME)

# Ensure the project working directory exists
os.makedirs(PROJECT_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
BEST_MODEL_PATH = os.path.join(PROJECT_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(PROJECT_DIR, "submission.csv")

# =============================================================================
# Data Configuration
# =============================================================================
# EfficientNet-B4 native resolution
IMAGE_SIZE = 380
NUM_CLASSES = 23
BATCH_SIZE = 32  # Adjusted for B4 memory footprint on A100
NUM_WORKERS = 12

# Debugging options to control dataset size
DEBUG = False
DEBUG_SAMPLE_SIZE = 2000

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = "efficientnet_b4"
PRETRAINED = True
USE_CONCAT_POOLING = True  # Use GAP + GMP

# =============================================================================
# Training Configuration
# =============================================================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Stage 1: Frozen Backbone, Train Head
LEARNING_RATE_STAGE1 = 1e-3
NUM_EPOCHS_STAGE1 = 5

# Stage 2: Unfreeze Top Blocks, Fine-tune
LEARNING_RATE_STAGE2 = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS_STAGE2 = 8

# Class Weights Strategy (Optional config if implemented in loss)
USE_CLASS_WEIGHTS = True


# =============================================================================
# Utility Functions
# =============================================================================
def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across all relevant libraries.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
