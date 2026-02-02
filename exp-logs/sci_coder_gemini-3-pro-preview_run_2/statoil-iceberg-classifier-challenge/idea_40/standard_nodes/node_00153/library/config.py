import os
import torch

# ==========================================
# PATH CONFIGURATION
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for the Quadrant-Context Wide-Body Network (QC-WBN)
WORKING_DIR = "./working/idea_40"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# DATA CONFIGURATION
# ==========================================
IMAGE_SIZE = 75
NUM_BANDS = 2
# Input channels: Band 1, Band 2, and (Band 1 + Band 2) / 2
NUM_CHANNELS = 3
# Use global min-max scaling statistics calculated from the training set
USE_GLOBAL_SCALING = True

# ==========================================
# MODEL CONFIGURATION
# ==========================================
# QC-WBN Architecture Hyperparameters
# Wide backbone filter count to prevent underfitting
NUM_FILTERS = 128
# High dropout rate to regularize the wide network
DROPOUT_RATE = 0.5
NUM_CLASSES = 1

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4  # Standard Adam learning rate
PATIENCE = 15  # Early stopping patience
SEED = 42  # Fixed seed for reproducibility

# ==========================================
# HARDWARE CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Number of subprocesses for data loading
