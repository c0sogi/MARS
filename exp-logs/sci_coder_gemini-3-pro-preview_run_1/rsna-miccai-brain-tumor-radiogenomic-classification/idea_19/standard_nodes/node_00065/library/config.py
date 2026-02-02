import os
import torch

# ==========================================
# Global Configuration & Hyperparameters
# ==========================================

# ------------------------------------------
# Hardware & Reproducibility
# ------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Adjust number of workers based on available vCPUs (12 available)
NUM_WORKERS = 8

# ------------------------------------------
# Paths & Directories
# ------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

WORKING_DIR = "./working"
# Specific directory for this experimental idea (Idea 19)
IDEA_DIR = os.path.join(WORKING_DIR, "idea_20")
CACHE_DIR = IDEA_DIR  # Directory to store cached numpy/parquet files

SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary writable directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ------------------------------------------
# Data Processing
# ------------------------------------------
IMG_SIZE = 224
# 3 Modalities (FLAIR, T1wCE, T2w) mapped to RGB channels
IN_CHANNELS = 3
MODALITIES = ["FLAIR", "T1wCE", "T2w"]

# ------------------------------------------
# Model Architecture
# ------------------------------------------
MODEL_NAME = "efficientnet_b0"
DROPOUT_RATE = 0.4
NUM_CLASSES = 1

# ------------------------------------------
# Training Loop
# ------------------------------------------
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive regularization for small dataset
N_FOLDS = 5
EARLY_STOPPING_PATIENCE = 5

# ------------------------------------------
# Development / Debugging
# ------------------------------------------
# Set DEBUG to True to run the pipeline on a small subset of data for quick testing
DEBUG = False
DEBUG_SAMPLE_SIZE = 32


def get_config_dict():
    """
    Returns the configuration as a dictionary for logging purposes.
    """
    return {k: v for k, v in globals().items() if k.isupper()}
