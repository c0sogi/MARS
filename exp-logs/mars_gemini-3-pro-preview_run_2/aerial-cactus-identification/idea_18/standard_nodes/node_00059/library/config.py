import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_18"
SUBMISSION_DIR = "./submission"

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
IMAGE_SIZE = (32, 32)
NUM_CHANNELS = 3
NUM_CLASSES = 1  # Binary classification

# =============================================================================
# MODEL ARCHITECTURE (Custom Wide ResNet)
# =============================================================================
# Backbone: Wide Channel Configuration [32, 64, 128] for 3 stages
# Stage 1: 32x32 -> 32 channels
# Stage 2: 16x16 -> 64 channels (Head A attached here)
# Stage 3: 8x8   -> 128 channels (Head B attached here)
BACKBONE_CHANNELS = [32, 64, 128]

# Deep Supervision
USE_DEEP_SUPERVISION = True
# Loss = 0.5 * Loss_HeadA (Stage 2) + 0.5 * Loss_HeadB (Stage 3)
LOSS_WEIGHTS = {"head_a": 0.5, "head_b": 0.5}

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 64
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # For AdamW
NUM_WORKERS = 2  # Number of dataloader workers

# =============================================================================
# ENSEMBLING & REPRODUCIBILITY
# =============================================================================
# Homogeneous Seed Averaging
SEEDS = [0, 1, 2, 3, 4]

# Compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DEBUGGING & DEVELOPMENT
# =============================================================================
# Set DEBUG to True to run on a small subset of data for quick verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 100


def setup_directories():
    """
    Creates the working and submission directories if they do not exist.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    print(f"Directories ensured: {WORKING_DIR}, {SUBMISSION_DIR}")
