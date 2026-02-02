import os
import torch

# =============================================================================
# Path Configuration
# =============================================================================
# Base Input Directory (Read-Only)
INPUT_DIR = "./input"

# Metadata Directory (Pre-generated CSVs)
METADATA_DIR = "./metadata"
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Working Directory for Idea 16
# All outputs (cache, checkpoints, submissions) must go here
WORKING_DIR = "./working/idea_16"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# Compute & Reproducibility
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of subprocesses for data loading
SEED = 42  # Fixed random seed for reproducibility

# =============================================================================
# Data Configuration
# =============================================================================
# Patch extraction settings
PATCH_SIZE = 128
SAMPLES_PER_IMAGE = 100  # High-density sampling: 100 random patches per image per epoch

# Image channels
IN_CHANNELS = 1  # Grayscale input
OUT_CHANNELS = 1  # Grayscale output (noise residual)

# =============================================================================
# Model Configuration
# =============================================================================
# Base capacity for the IC-ResUNet
BASE_FILTERS = 64

# =============================================================================
# Training Configuration
# =============================================================================
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2  # Aggressive regularization to prevent overfitting
EARLY_STOPPING_PATIENCE = 15

# =============================================================================
# Inference Configuration
# =============================================================================
OVERLAP_RATIO = 0.5  # 50% overlap for sliding window inference to reduce edge artifacts

# =============================================================================
# Debug / Development Flags
# =============================================================================
# Use these to limit dataset size for rapid testing/debugging
DEBUG = False
DEBUG_SAMPLE_SIZE = 10  # Number of images to use if DEBUG is True
