import os
import torch
import random
import numpy as np

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"

# Working directory for this specific experiment (Idea 41)
# We use a specific subdirectory to avoid conflicts with other runs
WORKING_DIR = "./working/idea_41"

# Metadata file paths (pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache file paths for deterministic loading
CACHE_TRAIN_X = os.path.join(WORKING_DIR, "cached_train_X.npy")
CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "cached_train_y.npy")
CACHE_VAL_X = os.path.join(WORKING_DIR, "cached_val_X.npy")
CACHE_VAL_Y = os.path.join(WORKING_DIR, "cached_val_y.npy")
CACHE_TEST_X = os.path.join(WORKING_DIR, "cached_test_X.npy")
CACHE_TEST_IDS = os.path.join(WORKING_DIR, "cached_test_ids.npy")

# ==========================================
# Data Configuration
# ==========================================
IMG_SIZE = 224
NUM_SLICES = 32  # Total slices sampled from the volume (High-Density)
SLICES_PER_VIEW = 16  # Slices per Siamese stream (Even/Odd split)
NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w

# Input channels for the backbone:
# Each view (Even/Odd) has 16 slices * 4 modalities = 64 channels
IN_CHANS = SLICES_PER_VIEW * NUM_MODALITIES

# ==========================================
# Model Configuration
# ==========================================
BACKBONE = "efficientnet_b0"
DROP_PATH_RATE = 0.2  # Stochastic Depth rate
NUM_CLASSES = 1

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15  # Sufficient for convergence with Early Stopping
PATIENCE = 5  # Early stopping patience
SEED = 42

# ==========================================
# Compute
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of data loading workers


# ==========================================
# Utility Functions
# ==========================================
def setup_directories():
    """
    Ensures that the necessary working and submission directories exist.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
