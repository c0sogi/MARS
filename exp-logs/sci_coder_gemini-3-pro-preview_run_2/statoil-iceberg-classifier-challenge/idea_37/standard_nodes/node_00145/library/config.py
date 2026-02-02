import os
import torch

# ==========================================
# 1. File Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Specific working directory for this idea (Idea 37)
# This serves as both the cache directory and the output directory for artifacts
CACHE_DIR = "./working/idea_37"
OUTPUT_DIR = CACHE_DIR

# Directory for final submission files
SUBMISSION_DIR = "./submission"

# Ensure necessary directories exist immediately upon import
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# 2. Global Configuration
# ==========================================
RANDOM_SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 3. Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10
NUM_FOLDS = 5

# ==========================================
# 4. Model Hyperparameters
# ==========================================
DROPOUT_RATE = 0.5
IMAGE_SIZE = 75
NUM_CLASSES = 1  # Binary classification (Ship vs Iceberg)
