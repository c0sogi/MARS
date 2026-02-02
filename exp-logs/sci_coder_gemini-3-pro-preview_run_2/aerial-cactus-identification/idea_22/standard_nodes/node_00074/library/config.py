import os
import torch

# ==========================================
# Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_22"
SUBMISSION_DIR = "./submission"

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission Output
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
IMAGE_SIZE = (32, 32)
NUM_CLASSES = 1
INPUT_CHANNELS = 3

# ==========================================
# Model Architecture Configuration
# ==========================================
# "Super-Wide" Channel Configuration as per Lesson 64
CHANNELS = [64, 128, 256]

# Res2Net Parameters
RES2NET_SCALE = 4  # 's' parameter: number of feature groups
RES2NET_BASE_WIDTH = 16  # approx CHANNELS[0] // RES2NET_SCALE, though dynamic in code

# Architecture Flags
USE_SE = True  # Enable Squeeze-and-Excitation
DROP_PATH_RATE = 0.0  # Stochastic depth (optional, kept 0 for stability)

# ==========================================
# Training Hyperparameters
# ==========================================
SEEDS = [0, 1, 2, 3, 4]  # Homogeneous Seed Averaging
EPOCHS = 20  # Reduced schedule due to fast convergence of Wide SE models
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2  # Standard for AdamW
ETA_MIN = 1e-6  # Minimum LR for Cosine Annealing

# ==========================================
# System Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
PIN_MEMORY = True


# ==========================================
# Utilities
# ==========================================
def setup_directories():
    """
    Ensures that the necessary working and submission directories exist.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
setup_directories()
