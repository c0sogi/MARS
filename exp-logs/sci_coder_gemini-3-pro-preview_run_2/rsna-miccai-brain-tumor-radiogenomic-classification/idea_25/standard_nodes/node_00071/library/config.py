import os
import torch

# -----------------------------------------------------------------------------
# Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching processed data and saving models
WORKING_DIR = "./working/idea_25"
os.makedirs(WORKING_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Preprocessing & Pipeline Configuration
# -----------------------------------------------------------------------------
IMG_SIZE = 224
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

# ROI Selection Parameters
ROI_DEPTH_MIN = 0.15
ROI_DEPTH_MAX = 0.85

# Stacking Logic
# We use 3 slices per stride group.
# Group 1: Local Context (Stride 2)
# Group 2: Broad Context (Stride 10)
NUM_SLICES_PER_GROUP = 3
STRIDE_LOCAL = 2
STRIDE_CONTEXT = 10

# Total Input Channels calculation:
# 4 Modalities * 3 Slices * 2 Strides = 24 Channels
INPUT_CHANNELS = len(MODALITIES) * NUM_SLICES_PER_GROUP * 2

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 1  # Binary classification (MGMT_value)

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # "Low Learning Rate"
WEIGHT_DECAY = 1e-2  # "Aggressive Weight Decay"
NUM_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 5

# -----------------------------------------------------------------------------
# System & Reproducibility
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# -----------------------------------------------------------------------------
# Debugging / Development
# -----------------------------------------------------------------------------
# Toggle DEBUG to True to run on a small subset of data for quick testing
DEBUG = False
MAX_DEBUG_SAMPLES = 50
