import os
import torch
import numpy as np
import random

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_68"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Input Files
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache Files (using idea-specific versioning)
CACHE_TRAIN = os.path.join(WORKING_DIR, "train_data_hi_gfdn_v1.npz")
CACHE_VAL = os.path.join(WORKING_DIR, "val_data_hi_gfdn_v1.npz")
CACHE_TEST = os.path.join(WORKING_DIR, "test_data_hi_gfdn_v1.npz")

# Output Files
MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# =============================================================================
# DATA SPECIFICATIONS
# =============================================================================
SEQ_LEN = 107
SCORED_LEN = 68

# The 3 columns used for the competition metric
TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

# All 5 ground truth columns provided in training data
ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

NUM_TARGETS = len(TARGET_COLS)

# =============================================================================
# MODEL HYPERPARAMETERS (HI-GFDN)
# =============================================================================
GROWTH_RATE = 32
LATENT_DIM = 64
RNN_HIDDEN = 64
FEEDBACK_DIM = 32
DROPOUT = 0.1
KERNEL_SIZE = 3
DILATIONS = [1, 2, 4, 8, 16, 32]

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-6
PATIENCE = 7  # Early stopping patience
NUM_WORKERS = 2
SEED = 42

# =============================================================================
# DEBUGGING & DEVELOPMENT
# =============================================================================
DEBUG = False
DEBUG_SUBSET_SIZE = 100  # Number of samples to use when DEBUG is True


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across numpy, random, and torch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def get_device():
    """
    Returns the available computing device (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
