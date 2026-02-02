import os
import random
import numpy as np
import torch

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

METADATA_DIR = "./metadata"
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

# Working directory for Idea Optimization (Sum Intensity ROI)
WORKING_DIR = "./working/idea_opt"
os.makedirs(WORKING_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Hyperparameters
# -----------------------------------------------------------------------------
IMG_SIZE = 224
NUM_SLICES = 3  # Anchor + neighbors
STRIDE = 5
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
NUM_MODALITIES = len(MODALITIES)
IN_CHANNELS = NUM_MODALITIES * NUM_SLICES  # 4 * 3 = 12 channels

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 15  # Default, can be overridden
NUM_WORKERS = 4
SEED = 42

# -----------------------------------------------------------------------------
# Compute
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def seed_everything(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
