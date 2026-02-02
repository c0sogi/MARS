import os
import torch
import random
import numpy as np

# ==========================================
# PATH CONFIGURATION
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths (Read-Only)
TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")

# Metadata Paths (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Caching & Output Paths
CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
os.makedirs(CACHE_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
MODEL_NAME = "mobilenet_v2"
NUM_CLASSES = 5270
IMG_SIZE = 224  # Standard input size for MobileNetV2
CHANNELS = 3

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
# Batch size optimized for A100 40GB VRAM
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 10  # Default epochs, can be overridden by early stopping
NUM_WORKERS = 8  # Utilizing available vCPUs
SEED = 42

# Compute Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
