import os
import random
import numpy as np
import torch

# ==========================================
# PATHS & DIRECTORIES
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
FEATURE_DIR = WORKING_DIR  # Directory to store cached parquet/npy files
SUBMISSION_DIR = "./submission"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
BACKBONE_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROJECTION_DIM = 512
TRANSFORMER_LAYERS = 2
N_HEADS = 4
MAX_ANCHOR_SEQ_LEN = (
    1024  # Max number of code cells (anchors) to handle in the transformer
)
DROPOUT = 0.1

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
BATCH_SIZE = 64
LR = 1e-3
NUM_EPOCHS = 5
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# SYSTEM SETUP
# ==========================================
SEED = 42


def setup_system(seed=SEED):
    """
    Sets fixed random seeds for reproducibility and ensures directories exist.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Re-verify directories in case function is called before module load completes
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Set environment variables for deterministic behavior where possible
    os.environ["PYTHONHASHSEED"] = str(seed)
