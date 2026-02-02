import os
import random
import numpy as np
import torch
from pathlib import Path

# --- Paths ---
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
WORKING_DIR = Path("./working")
CACHE_DIR = WORKING_DIR / "idea_1"
SUBMISSION_PATH = Path("submission/submission.csv")

# Ensure mutable directories exist
WORKING_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Data Constants ---
Z_DIM = 65  # Number of slices in the z-direction
PATCH_SIZE = 256  # Height/Width of patches used for training
INFERENCE_STRIDE = (
    224  # Stride for tiling during inference (overlap = PATCH_SIZE - STRIDE)
)

# --- Training Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 15  # Fixed number of epochs for baseline strategy
NUM_WORKERS = 4  # Number of subprocesses for data loading
SEED = 42  # Fixed random seed

# --- Inference Constants ---
THRESHOLD = 0.5  # Probability threshold for binary classification

# --- Compute Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Utilities ---
def seed_everything(seed: int = SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
