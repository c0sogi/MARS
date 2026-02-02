import os
import torch

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# Directory & File Paths
# ==========================================
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_23"
SUBMISSION_DIR = "./submission"

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Processing Hyperparameters
# ==========================================
IMG_SIZE = 224
STRIDE = 5  # The delta for volumetric packing (z - delta, z, z + delta)
IN_CHANNELS = 9  # 3 modalities (FLAIR, T1wCE, T2w) * 3 depths

# Debugging / Development
# Set DEBUG to True to use a smaller subset of data for rapid testing
DEBUG = False
DEBUG_SAMPLE_SIZE = 50

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive weight decay as per AGIV strategy
DROPOUT_RATE = 0.3
PATIENCE = 5  # Early stopping patience
N_FOLDS = 5  # Number of folds for Cross-Validation

# ==========================================
# Compute Resources
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of subprocesses for data loading


def get_config_dict():
    """
    Returns the configuration as a dictionary for logging or dynamic usage.
    """
    return {
        "SEED": SEED,
        "IMG_SIZE": IMG_SIZE,
        "STRIDE": STRIDE,
        "IN_CHANNELS": IN_CHANNELS,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_EPOCHS": NUM_EPOCHS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DROPOUT_RATE": DROPOUT_RATE,
        "PATIENCE": PATIENCE,
        "N_FOLDS": N_FOLDS,
        "DEVICE": DEVICE,
        "DEBUG": DEBUG,
    }
