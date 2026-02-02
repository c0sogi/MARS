import os
import torch
import random
import numpy as np

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Input Data Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata File Paths (Generated in previous steps)
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

# Working Directory for Idea 4 (Parallel-Scale Dilated CNN)
WORKING_DIR = "./working/idea_4"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
PREDICTION_DIR = os.path.join(WORKING_DIR, "predictions")

# Submission Output
SUBMISSION_PATH = "./submission.csv"

# =============================================================================
# DATA PARAMETERS
# =============================================================================

NUM_SLICES = 65
PATCH_SIZE = (512, 512)
INPUT_SHAPE = (NUM_SLICES, PATCH_SIZE[0], PATCH_SIZE[1])

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Parallel-Scale Dilated CNN Configuration
IN_CHANNELS = NUM_SLICES
STEM_CHANNELS = 64
# Width of the parallel branches (Texture, Stroke, Context)
# Kept moderate to allow Batch Size = 32 within memory constraints
BRANCH_CHANNELS = 16

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
POS_WEIGHT = 2.0  # Weight for positive class in BCEWithLogitsLoss
NUM_WORKERS = 2  # Number of dataloader workers

# =============================================================================
# COMPUTE CONFIGURATION
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def setup_directories():
    """
    Creates the necessary directory structure for the current idea.
    """
    dirs = [WORKING_DIR, CACHE_DIR, CHECKPOINT_DIR, PREDICTION_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


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


# Initialize environment on import
setup_directories()
set_seed()
