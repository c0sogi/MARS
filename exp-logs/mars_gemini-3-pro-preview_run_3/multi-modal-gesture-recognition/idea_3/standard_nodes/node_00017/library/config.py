import os
import torch
import numpy as np
import random

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Data Processing Hyperparameters
# ==========================================
# Sliding Window Strategy
WINDOW_SIZE = 128  # Length of sequence window in frames
STRIDE = 64  # Step size for sliding window (50% overlap)

# Post-processing
MEDIAN_FILTER_KERNEL = 7  # Kernel size for median filtering (must be odd)

# Feature Dimensions
# Skeleton: 20 joints * 3 coordinates (x,y,z) = 60
# Velocity: 20 joints * 3 coordinates = 60
# Audio: 13 MFCC coefficients
SKELETON_DIM = 60
VELOCITY_DIM = 60
AUDIO_DIM = 13
INPUT_DIM = SKELETON_DIM + VELOCITY_DIM + AUDIO_DIM  # Total: 133

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
# Stage 1: Bi-GRU Encoder
GRU_HIDDEN_DIM = 128
GRU_NUM_LAYERS = 2
DROPOUT = 0.3

# Stage 2: TCN Refinement
TCN_NUM_CHANNELS = [64, 64, 64, 64]  # Channel depth for TCN layers
TCN_KERNEL_SIZE = 3
TCN_DROPOUT = 0.2

# Classes: 20 Gestures + 1 Background
NUM_CLASSES = 21

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # Early stopping patience

# Loss Weights
BG_CLASS_WEIGHT = 0.2  # Weight for background class (0) to handle imbalance
SMOOTHING_LOSS_WEIGHT = 0.15  # Weight for TCN smoothing loss (MSE on log-probs)

# ==========================================
# Reproducibility
# ==========================================
RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    """Sets the random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# Label Mappings
# ==========================================
# Raw label map from dataset description
RAW_LABEL_MAP = {
    "vattene": 1,
    "vieniqui": 2,
    "perfetto": 3,
    "furbo": 4,
    "cheduepalle": 5,
    "chevuoi": 6,
    "daccordo": 7,
    "seipazzo": 8,
    "combinato": 9,
    "freganiente": 10,
    "ok": 11,
    "cosatifarei": 12,
    "basta": 13,
    "prendere": 14,
    "noncenepiu": 15,
    "fame": 16,
    "tantotempo": 17,
    "buonissimo": 18,
    "messidaccordo": 19,
    "sonostufo": 20,
}

# Inverse map for decoding predictions
# Maps ID -> Name. Includes 0 for 'background'.
ID_TO_NAME = {v: k for k, v in RAW_LABEL_MAP.items()}
ID_TO_NAME[0] = "background"
