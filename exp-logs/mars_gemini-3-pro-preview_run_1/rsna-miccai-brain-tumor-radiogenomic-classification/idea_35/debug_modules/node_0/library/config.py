import os
import torch

# ==========================================
# System & Reproducibility
# ==========================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# ==========================================
# File Paths & Directories
# ==========================================
# Input Directories
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata Paths (Pre-generated)
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Working Directory for Caching and Checkpoints
# Using specific idea folder as requested
WORKING_DIR = "./working/idea_35"
os.makedirs(WORKING_DIR, exist_ok=True)

CACHE_DIR = WORKING_DIR  # Alias for clarity in data processing modules

# Output Paths
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Processing Hyperparameters
# ==========================================
IMG_SIZE = 224
NUM_CHANNELS = 9  # 3 modalities (FLAIR, T1wCE, T2w) * 3 depths
RELATIVE_DEPTHS = [0.4, 0.5, 0.6]  # Scale-Invariant Relative Depth Sampling

# ==========================================
# Model Hyperparameters (SICAV Network)
# ==========================================
BACKBONE = "efficientnet_b0"
DROPOUT_RATE = 0.3
NUM_CLASSES = 1

# ==========================================
# Training Hyperparameters
# ==========================================
NUM_FOLDS = 5
BATCH_SIZE = 32  # Safe for A100 40GB with 224x224x9 input
EPOCHS = 20  # Sufficient for convergence with early stopping
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive regularization as per strategy
PATIENCE = 5  # For Early Stopping

# ==========================================
# Augmentation Settings
# ==========================================
# Note: Translation/Shift and Scaling are strictly excluded to preserve
# the spatial priors established by centroid alignment.
AUG_ROTATION_LIMIT = 15
AUG_ELASTIC_ALPHA = 1.0
AUG_GRID_DISTORT_LIMIT = 0.3
