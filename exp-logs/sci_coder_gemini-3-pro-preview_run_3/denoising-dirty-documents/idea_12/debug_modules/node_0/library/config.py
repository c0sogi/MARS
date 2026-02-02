import os
import torch
import random
import numpy as np

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_12"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Data Processing Configuration
# ==========================================
PATCH_SIZE = 50
# Stage 1: Large stride for sparse sampling (fast convergence)
STRIDE_SPARSE = 20
# Stage 2: Small stride for dense sampling (high capacity refinement)
STRIDE_DENSE = 5

NUM_WORKERS = 4

# ==========================================
# Model Architecture Configuration (CA-ResDnCNN)
# ==========================================
N_CHANNELS = 1  # Input is grayscale
N_FEATS = 64  # Number of feature maps in hidden layers
N_RES_BLOCKS = 24  # Number of residual blocks in the deep stack
KERNEL_SIZE = 3  # Spatial size of convolution filters
REDUCTION = 16  # Reduction ratio for Coordinate Attention mechanism

# ==========================================
# Training Configuration
# ==========================================
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Curriculum Training Schedule
# Stage 1: Initial convergence on sparse data
NUM_EPOCHS_STAGE_1 = 50
# Stage 2: Refinement on dense data (utilizing remaining runtime)
NUM_EPOCHS_STAGE_2 = 100

PATIENCE = 10  # For early stopping

# ==========================================
# System Configuration
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to the global SEED constant.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
