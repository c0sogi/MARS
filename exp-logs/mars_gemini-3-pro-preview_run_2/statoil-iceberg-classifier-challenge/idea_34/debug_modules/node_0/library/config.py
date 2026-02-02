import os
import torch
import random
import numpy as np

# ==========================================
# PATH CONFIGURATION
# ==========================================
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_34"
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Paths
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Paths
TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

# Artifact Paths
CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")


def get_model_path(fold_idx):
    """Returns the file path for saving the model of a specific fold."""
    return os.path.join(WORKING_DIR, f"model_fold_{fold_idx}.pth")


# ==========================================
# HYPERPARAMETERS
# ==========================================
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DROPOUT_RATE = 0.5
NUM_FOLDS = 5
PATIENCE = 10  # Early stopping patience

# ==========================================
# MODEL SPECIFICS
# ==========================================
IMAGE_SIZE = 75
NUM_CHANNELS = 3  # Band 1, Band 2, Mean(B1, B2)
GEM_P_INIT = 3.0  # Initial p-value for Generalized Mean Pooling

# ==========================================
# COMPUTE DEVICE
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# UTILITIES
# ==========================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
