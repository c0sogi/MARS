import os
import torch
import random
import numpy as np

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Specific cache directory for this idea (Idea 21)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_21")
os.makedirs(CACHE_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# ==========================================
# Data Configuration
# ==========================================
IMAGE_SIZE = 256
NUM_CLASSES = 1

# Input Engineering Strategy:
# 3 Channels (Ash Composite) + 3 Channels (Temporal Diff) + 2 Channels (Spatial Coords)
INPUT_CHANNELS = 8

# Band Definitions
# Bands used for Ash Color Scheme (Bands 11, 14, 15)
ASH_BAND_IDS = [11, 14, 15]
# Bands used for Temporal Difference (t=4 - t=3)
DIFF_BAND_IDS = [11, 14, 15]

# ==========================================
# Training Hyperparameters
# ==========================================
# Batch size of 32 is critical for statistical stability of Batch-Level Dice loss
BATCH_SIZE = 32

# Train for 30 epochs to avoid underfitting fine-grained details
EPOCHS = 30

# Optimization
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 2  # Adjusted for available vCPUs (12 total, safe per loader)

# ==========================================
# Compute Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# ==========================================
# Utility Functions
# ==========================================
def seed_everything(seed: int = SEED):
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
