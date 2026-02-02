import os
import torch
import numpy as np
import random


def seed_everything(seed: int = 42):
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


# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

METADATA_DIR = "./metadata"
TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
VAL_META = os.path.join(METADATA_DIR, "val.csv")
TEST_META = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching processed data and saving models
WORKING_DIR = "./working/optimized"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
SEED = 42
IMG_SIZE = 512
BATCH_SIZE = 16  # Conservative batch size for 512x512 on A100
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4  # Optimized for 12 vCPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
ENCODER_NAME = "resnet18"
NUM_CLASSES = 4
CLASS_LABELS = [
    "Negative for Pneumonia",
    "Typical Appearance",
    "Indeterminate Appearance",
    "Atypical Appearance",
]

# =============================================================================
# TRAINING STRATEGY
# =============================================================================
# Loss Weighting: 1.0 for Classification, 10.0 for Segmentation
LOSS_WEIGHTS = {"class": 1.0, "seg": 10.0}

# Debugging / Development Flags
DEBUG = False
DEBUG_SAMPLE_SIZE = 100  # Number of samples to use when DEBUG is True

# Initialize environment
seed_everything(SEED)
