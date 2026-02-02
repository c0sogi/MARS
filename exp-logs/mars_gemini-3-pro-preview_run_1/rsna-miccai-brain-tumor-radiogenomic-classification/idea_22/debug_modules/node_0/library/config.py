import os
import torch

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Paths & Directories
# ==========================================
# Input Data (Read-Only)
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Working Directory (Write Access)
# Used for caching processed arrays and saving models
WORKING_DIR = "./working/idea_22"
os.makedirs(WORKING_DIR, exist_ok=True)

# Output Submission
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Processing Parameters
# ==========================================
# Image Dimensions for EfficientNet-B0
IMG_SIZE = 224

# Volumetric Sampling
# We take the center slice (z) and slices at (z - stride) and (z + stride)
SLICE_STRIDE = 5

# Modalities to use (Order matters for channel stacking)
# Per AC-WIV design: FLAIR, T1wCE, T2w (T1w is excluded)
MODALITIES = ["FLAIR", "T1wCE", "T2w"]

# ==========================================
# Model Architecture
# ==========================================
BACKBONE = "efficientnet_b0"
PRETRAINED = True

# Input Channels: 3 depths * 3 modalities = 9 channels
INPUT_CHANNELS = 9

# Regularization
DROPOUT_RATE = 0.3

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

# Cross-Validation
NUM_FOLDS = 5

# ==========================================
# Compute Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of subprocesses for data loading
NUM_WORKERS = 4
