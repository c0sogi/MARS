import os
import torch
import random
import numpy as np

# --- Paths ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
OUTPUT_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# --- Data Parameters ---
# Input dimensions based on dataset description: (6, 273, 256)
# We concatenate vertically: 273 * 6 = 1638
IMAGE_HEIGHT = 1638
IMAGE_WIDTH = 256
IN_CHANNELS = 1
NUM_CLASSES = 1

# --- Model Parameters ---
MODEL_ARCH = "resnet18"  # Base architecture

# --- Training Hyperparameters ---
SEED = 42
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
MAX_LR = 1e-2  # For OneCycleLR scheduler
PATIENCE = 3  # For Early Stopping

# --- Debugging ---
# Set DEBUG to True to run on a small subset of data for testing pipeline
DEBUG = False
DEBUG_SAMPLE_SIZE = 500

# --- Compute ---
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Automatically set seed on import
set_seed(SEED)
