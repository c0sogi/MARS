import os
import torch

# =============================================================================
# FILE PATHS & DIRECTORIES
# =============================================================================
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output File Paths
# Template for saving model checkpoints (e.g., model_seed_0.pth)
MODEL_CHECKPOINT_TEMPLATE = os.path.join(WORKING_DIR, "model_seed_{}.pth")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
IMAGE_SIZE = (32, 32)
IN_CHANNELS = 3
NUM_CLASSES = 1  # Binary classification

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# Ensembling: Homogeneous Seed Averaging with 5 instances
SEEDS = [0, 1, 2, 3, 4]

# Optimization (AdamW + Cosine Annealing)
EPOCHS = 25
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01  # Standard for AdamW
EARLY_STOPPING_PATIENCE = 5

# Scheduler Settings (CosineAnnealingLR)
T_MAX = EPOCHS
ETA_MIN = 1e-6

# =============================================================================
# COMPUTE CONFIGURATION
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Optimized for 12 vCPUs
NUM_WORKERS = 4
PIN_MEMORY = True
