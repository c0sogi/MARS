import os
import torch

# -----------------------------------------------------------------------------
# File System Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for caching processed data and saving models
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_1")
CHECKPOINT_PATH = os.path.join(IDEA_DIR, "best_model.pth")

# Submission directory
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary writable directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Configuration
# -----------------------------------------------------------------------------
IMG_SIZE = 256
NUM_SLICES = 3  # Extracting slices at 25%, 50%, 75% depth
NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w
IN_CHANNELS = NUM_SLICES * NUM_MODALITIES  # 3 slices * 4 modalities = 12 channels

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 15
PATIENCE = 5  # For early stopping

# -----------------------------------------------------------------------------
# Compute Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Adjust based on vCPU availability (12 vCPUs available)
