import os
import torch
from pathlib import Path

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_DIR = Path("./input")
METADATA_DIR = Path("./metadata")
WORKING_DIR = Path("./working")
CACHE_DIR = WORKING_DIR / "idea_1"
SUBMISSION_DIR = Path("./submission")

# Specific Metadata Files
TRAIN_META_PATH = METADATA_DIR / "train.csv"
VAL_META_PATH = METADATA_DIR / "val.csv"
TEST_META_PATH = METADATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = INPUT_DIR / "sample_submission.csv"

# Ensure necessary writable directories exist
WORKING_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Data Specifications
# ==========================================
# Based on EDA: 32093 unique categories in training set
NUM_CLASSES = 32093
# Standard input size for ResNet-18
IMG_SIZE = 224
NUM_CHANNELS = 3

# ==========================================
# Model & Training Hyperparameters
# ==========================================
MODEL_NAME = "resnet18"
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 4
WEIGHT_DECAY = 1e-4

# Reproducibility
SEED = 42

# Hardware
# We have 12 vCPUs, so 4-8 workers is usually optimal
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Debugging / Development
# ==========================================
# Set to True to run on a small subset of data for quick pipeline verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 2000
