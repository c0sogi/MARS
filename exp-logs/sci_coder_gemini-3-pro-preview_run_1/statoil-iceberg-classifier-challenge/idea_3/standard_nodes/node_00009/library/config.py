import os
import random
import numpy as np
import torch

# ====================================================
# Directory Setup
# ====================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ====================================================
# File Paths
# ====================================================
# Raw Data
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ====================================================
# Hyperparameters
# ====================================================
# Data
IMG_SIZE = 224
CHANNELS = 3  # Band 1, Band 2, Mean
NUM_CLASSES = 1

# Training
SEED = 42
N_FOLDS = 5
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.05
PATIENCE = 5  # For Early Stopping
NUM_WORKERS = 4

# Model
MODEL_NAME = "resnet18"
DROPOUT_RATE = 0.5  # For the classification head

# Debugging
DEBUG = False
DEBUG_SAMPLES = 100


# ====================================================
# Utility Functions
# ====================================================
def get_model_path(fold_idx):
    """
    Returns the file path for saving/loading the model checkpoint for a specific fold.

    Args:
        fold_idx (int): The index of the current fold (0-based).

    Returns:
        str: The full path to the model checkpoint file.
    """
    return os.path.join(WORKING_DIR, f"{MODEL_NAME}_fold_{fold_idx}.pth")


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
