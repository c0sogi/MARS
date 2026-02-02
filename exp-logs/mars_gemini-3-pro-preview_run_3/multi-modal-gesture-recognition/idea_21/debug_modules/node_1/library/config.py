import os
import torch
import numpy as np
import random

# ==========================================
# 1. Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_21"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# 2. Data Configuration
# ==========================================
WINDOW_SIZE = 64
STRIDE = 32
NUM_CLASSES = 21  # 20 gestures + 1 background (Class 0)
BACKGROUND_CLASS_ID = 0

# Skeleton Configuration
NUM_JOINTS = 20
NUM_CHANNELS = 3  # x, y, z
USE_VELOCITY = True
USE_ACCELERATION = True

# Audio Configuration
NUM_MFCC = 13

# Input Dimension Calculation
# Features: Position (20*3) + Velocity (20*3) + Acceleration (20*3) + MFCC (13)
# Total: 60 + 60 + 60 + 13 = 193
INPUT_DIM = (NUM_JOINTS * NUM_CHANNELS) * (
    1 + int(USE_VELOCITY) + int(USE_ACCELERATION)
) + NUM_MFCC

# ==========================================
# 3. Model Architecture Configuration
# ==========================================
# Stage 1: High-Capacity Kinematic Sequence Encoder (Bi-GRU)
GRU_HIDDEN_SIZE = 128  # Bidirectional implies output dim = 256
GRU_LAYERS = 1
GRU_DROPOUT = 0.3

# Stage 2 & 3: Hierarchical Sawtooth Refinement (MS-TCN)
MSTCN_FEATURES = 64
MSTCN_KERNEL_SIZE = 3
MSTCN_DROPOUT = 0.3

# Hierarchical Sawtooth Dilation Schedule
# 3 blocks of [1, 2, 4] to balance local resolution and receptive field
SAWTOOTH_DILATIONS = [1, 2, 4, 1, 2, 4, 1, 2, 4]

# ==========================================
# 4. Training Configuration
# ==========================================
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Loss Function Weights
BACKGROUND_WEIGHT = 0.2  # Weight for class 0
SMOOTHING_LOSS_WEIGHT = 0.15
SMOOTHING_THRESHOLD = 1.0  # Truncated MSE threshold

# Debugging / Development
DEBUG = False
DEBUG_SUBSET_SIZE = 50  # Number of samples to use when DEBUG is True


# ==========================================
# 5. Utilities
# ==========================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the appropriate torch device (CUDA or CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
