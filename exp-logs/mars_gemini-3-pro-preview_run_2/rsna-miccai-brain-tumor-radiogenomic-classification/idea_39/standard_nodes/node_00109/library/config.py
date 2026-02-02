import os
import torch

# -----------------------------------------------------------------------------
# Global Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

METADATA_DIR = "./metadata"
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching processed data and saving model artifacts
# We ensure this directory exists as per requirements
WORKING_DIR = "./working/idea_39"
os.makedirs(WORKING_DIR, exist_ok=True)

CACHE_DIR = WORKING_DIR
MODEL_SAVE_DIR = WORKING_DIR
SUBMISSION_PATH = "./submission/submission.csv"

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
IMG_SIZE = 224
NUM_CHANNELS = 12  # 4 modalities (FLAIR, T1w, T1wCE, T2w) * 3 slices per stack

# Single Model Stride (Context)
STRIDE = 5

# Fidelity-Aligned ROI Selection
ROI_DEPTH_MIN = 0.15
ROI_DEPTH_MAX = 0.85

# Augmentation
AUG_ROTATE_DEG = 15
AUG_PROB = 0.5

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "efficientnet_b0"
STEM_GROUPS = 4  # For Asymmetric Grouped Convolutions in the stem
DROPOUT_RATE = 0.5  # Regularization before the final head

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive weight decay as per idea

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# Debugging / Development
# Set to an integer (e.g., 50) to limit dataset size for rapid debugging.
# Set to None for full training.
DEBUG_SAMPLE_SIZE = None


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
