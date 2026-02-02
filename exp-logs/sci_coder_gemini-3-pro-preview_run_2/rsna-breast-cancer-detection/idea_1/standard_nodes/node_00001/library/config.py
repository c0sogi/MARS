import os
import random
import numpy as np
import torch

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

# Ensure writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# Data Configuration
# ==========================================
IMG_SIZE = 512
NUM_WORKERS = 4  # Number of subprocesses for data loading

# ==========================================
# Model & Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 5
POS_WEIGHT = 50.0  # Weight for positive class in loss function to handle imbalance

# ==========================================
# Reproducibility & Compute
# ==========================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for various libraries to ensure reproducibility.

    Args:
        seed (int): The random seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
