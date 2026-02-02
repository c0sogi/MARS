import os
import torch
import random
import numpy as np

# ==========================================
# Global Configuration Constants
# ==========================================

# ------------------------------------------
# 1. File System Paths
# ------------------------------------------
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

# Data File Paths (Using Metadata Splits)
TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
CACHE_DIR = WORKING_DIR  # Directory for caching parquet/npy files

# ------------------------------------------
# 2. Data & Feature Configuration
# ------------------------------------------
SEED = 42
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# Feature definitions
# f_27 is the string feature to be decomposed
STRING_COL = "f_27"
# Discrete features to be treated as categorical
DISCRETE_COLS = ["f_29", "f_30"]
# Target column name
TARGET_COL = "target"
# ID column name
ID_COL = "id"

# ------------------------------------------
# 3. Model Architecture Hyperparameters
# ------------------------------------------
# Dual-Stream Funnel MLP Settings
EMBED_DIM = 16  # Fixed embedding dimension for categorical features
BACKBONE_LAYERS = [512, 256, 128]  # Funnel structure
DROPOUT = 0.2  # Dropout rate
OUTPUT_DIM = 1  # Binary classification

# ------------------------------------------
# 4. Training Hyperparameters
# ------------------------------------------
BATCH_SIZE = 1024
EPOCHS = 30
MAX_LR = 1e-3  # Max learning rate for OneCycleLR
WEIGHT_DECAY = 1e-5  # Calibrated for AdamW
PATIENCE = 5  # Early stopping patience

# ------------------------------------------
# 5. Debugging & Control
# ------------------------------------------
# Set to True to run on a small subset of data for testing pipeline
DEBUG = False
# If DEBUG is True, use this many samples
MAX_SAMPLES = 5000

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Setup Utilities
# ==========================================


def setup_environment(seed=SEED):
    """
    Initializes the environment by creating necessary directories and
    setting random seeds for reproducibility.
    """
    # Create directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN if needed (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seeding
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Environment setup complete. Device: {DEVICE}, Seed: {seed}")
    print(f"Working Directory: {WORKING_DIR}")
