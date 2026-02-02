import os
import torch

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_15"
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Processing Hyperparameters
# ==========================================
# Input Dimensions
IMAGE_SIZE = 224

# Slab Construction Logic (Standard 2.5D)
SLICE_DEPTH = 1  # Number of consecutive slices per modality (Just the middle one)
NUM_SLABS = 1  # Number of slabs extracted per subject (Just the median)
SLAB_STRIDE = 0  # Irrelevant for single slab

# Channel Configuration
# 3 Modalities (FLAIR, T1wCE, T2w) * 1 Slice/Modality = 3 Channels (RGB-like)
IN_CHANNELS = 3

# ==========================================
# Model Hyperparameters
# ==========================================
BACKBONE_NAME = "efficientnet_b0"
DROPOUT_RATE = 0.3
NUM_CLASSES = 1

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive weight decay for small dataset
NUM_EPOCHS = 15  # Sufficient for convergence with pre-trained backbone
NUM_FOLDS = 5  # For GroupKFold

# ==========================================
# Compute & Hardware
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
