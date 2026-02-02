import os
import torch

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Specific File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
SEED = 42
WINDOW_SIZE = 64
STRIDE = 32  # 50% overlap for sliding windows during training
NUM_CLASSES = 21  # 20 gesture classes + 1 background class (index 0)

# Skeleton Feature Configuration
# 20 joints * 3 coordinates (X, Y, Z)
SKELETON_JOINTS = 20
SKELETON_COORDS = 3
# Features: Position (3) + Velocity (3) + Acceleration (3) = 9 features per joint
SKELETON_FEAT_PER_JOINT = 9
INPUT_DIM_SKELETON = SKELETON_JOINTS * SKELETON_FEAT_PER_JOINT  # 180

# Audio Feature Configuration
AUDIO_SR = 16000
N_MFCC = 13
INPUT_DIM_AUDIO = N_MFCC

# Total Input Dimension for Early Fusion
INPUT_DIM = INPUT_DIM_SKELETON + INPUT_DIM_AUDIO  # 193

# ==========================================
# Model Architecture Configuration
# ==========================================
# Stage 1: Sequence Encoder (Bi-GRU)
GRU_HIDDEN_SIZE = 128
GRU_NUM_LAYERS = 2
GRU_DROPOUT = 0.3

# Stage 2 & 3: Refinement Modules (Dilated TCN)
# Channels for the causal dilated convolutions
TCN_NUM_CHANNELS = [64, 64, 64, 64]
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.2

# ==========================================
# Training Configuration
# ==========================================
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # For Early Stopping

# Loss Function Configuration
# Background class (0) is weighted down to 0.2 to handle imbalance
BACKGROUND_WEIGHT = 0.2
CLASS_WEIGHTS = torch.ones(NUM_CLASSES)
CLASS_WEIGHTS[0] = BACKGROUND_WEIGHT

# Coefficient for the Log-Space Smoothing Loss (Truncated MSE)
SMOOTHING_LAMBDA = 0.15

# ==========================================
# Hardware & Execution Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4


def get_device():
    """Returns the torch device."""
    return torch.device(DEVICE)
