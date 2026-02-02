import os
import torch
import random
import numpy as np

# ==========================================
# Path Configuration
# ==========================================
INPUT_ROOT = "./input"
METADATA_DIR = "./metadata"

# Metadata CSV paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Output Directories
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths for artifacts
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "resnet18_best.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
LABEL_ENCODER_PATH = os.path.join(WORKING_DIR, "label_encoder.json")

# ==========================================
# Model Configuration
# ==========================================
NUM_CLASSES = 32093
IMG_SIZE = 224
ARCH = "resnet18"

# ==========================================
# Training Configuration
# ==========================================
BATCH_SIZE = 256  # Fits comfortably in A100 40GB
LEARNING_RATE = 1e-3  # Standard starting LR for Adam/ResNet
NUM_EPOCHS = 20  # Maximum number of epochs
PATIENCE = 3  # Early stopping patience
NUM_WORKERS = 8  # Number of dataloader workers (12 vCPUs available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ==========================================
# Debug / Development Configuration
# ==========================================
# Set these to an integer (e.g., 1000) to limit dataset size for quick debugging
# Set to None to use the full dataset
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None


# ==========================================
# Utility Functions
# ==========================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
