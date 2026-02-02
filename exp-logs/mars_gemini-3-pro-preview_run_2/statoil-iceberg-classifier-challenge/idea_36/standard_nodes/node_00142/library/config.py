import os
import torch

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_36"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# 2. DATA CONFIGURATION
# ==========================================
# Image dimensions
IMAGE_HEIGHT = 75
IMAGE_WIDTH = 75
NUM_BANDS = 2  # HH, HV
INPUT_CHANNELS = 3  # HH, HV, Average

# Metadata files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw data files (referenced by metadata)
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# ==========================================
# 3. HYPERPARAMETERS
# ==========================================
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_FOLDS = 5
NUM_EPOCHS = 50  # Max epochs, controlled by early stopping
PATIENCE = 10  # Early stopping patience

# Model specific
DROPOUT_RATE = 0.5
BACKBONE_FILTERS = 128
NUM_CLASSES = 1  # Binary classification

# Compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2  # Adjust based on available vCPUs


# ==========================================
# 4. UTILITIES
# ==========================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
