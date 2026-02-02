import os
import random
import numpy as np
import torch

# =============================================================================
# Configuration & Constants
# =============================================================================

# Random Seed
SEED = 42

# Image Parameters
IMG_SIZE = 224
CHANNELS = 3

# Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_4")

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Files
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Hyperparameters & Debugging
# These control dataset size (for debugging) and training steps
DEBUG = False
DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True
BATCH_SIZE = 32  # Batch size for data loaders
NUM_EPOCHS = 10  # Default number of epochs (if applicable)
NUM_WORKERS = 4  # Number of data loading workers

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(IDEA_DIR, exist_ok=True)

# =============================================================================
# Utility Functions
# =============================================================================


def set_seed(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """
    Detects and returns the available PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
