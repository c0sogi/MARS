import os
import random
import numpy as np
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

# Sample Submission Path
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sampleSubmission.csv")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
IMG_SIZE = 160
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 1000
N_FOLDS = 5
SEED = 42

# =============================================================================
# HARDWARE & COMPUTE
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of workers for data loading (adjust based on vCPUs)
NUM_WORKERS = 4


# =============================================================================
# UTILITIES
# =============================================================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to the global SEED constant.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
