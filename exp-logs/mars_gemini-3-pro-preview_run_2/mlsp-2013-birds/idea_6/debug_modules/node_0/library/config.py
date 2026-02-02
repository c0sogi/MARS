import os
import random
import numpy as np
import torch

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Reproducibility
# ==========================================
SEED = 42


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# Data Configuration
# ==========================================
# Input dimensions: 224 (Height) x 512 (Width) to preserve temporal resolution
IMG_HEIGHT = 224
IMG_WIDTH = 512

# Input channels: 3 (Spectrogram, Delta, Delta-Delta)
CHANNELS = 3

# Number of bird species
NUM_CLASSES = 19

# DataLoader settings
NUM_WORKERS = 4

# ==========================================
# Model Configuration
# ==========================================
BACKBONE = "resnet34"
PRETRAINED = True

# Multi-Sample Dropout settings (dropout rates for parallel branches)
DROPOUT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4]

# ==========================================
# Training Configuration
# ==========================================
NUM_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Mixup Regularization
MIXUP_ALPHA = 0.4

# Early Stopping
EARLY_STOPPING_PATIENCE = 10

# ==========================================
# Debugging
# ==========================================
# Set DEBUG to True to run on a small subset of data for testing the pipeline
DEBUG = False
DEBUG_SUBSET_SIZE = 20
