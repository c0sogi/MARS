import os
import torch
import random
import numpy as np

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Working Directory for Caching
WORKING_DIR = "./working/idea_33"
os.makedirs(WORKING_DIR, exist_ok=True)

# ==========================================
# Data Configuration (RARV Strategy)
# ==========================================
IMG_SIZE = (224, 224)

# The specific modalities used in the 9-channel input
# Note: T1w is excluded to fit the 3-depth x 3-modality structure
MODALITIES = ["FLAIR", "T1wCE", "T2w"]

# Relative depths for sampling within the Brain ROI
# 0.4 = 40%, 0.5 = Center, 0.6 = 60%
ROI_DEPTHS = [0.4, 0.5, 0.6]

# Total input channels = len(MODALITIES) * len(ROI_DEPTHS) = 9
INPUT_CHANNELS = len(MODALITIES) * len(ROI_DEPTHS)

# ==========================================
# Model Hyperparameters
# ==========================================
BACKBONE = "efficientnet_b0"
NUM_CLASSES = 1
DROPOUT_RATE = 0.3  # Classifier dropout
INPUT_DROPOUT_PROB = 0.2  # Structured input dropout for RARV

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive regularization
NUM_WORKERS = 4
SEED = 42

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Utility Functions
# ==========================================
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
