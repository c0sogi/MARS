import os
import random
import numpy as np
import torch
import pandas as pd

# -----------------------------------------------------------------------------
# Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Global Hyperparameters
# -----------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# Data Parameters
IMAGE_SIZE = 256
BATCH_SIZE = 32
DEBUG = False  # Set to True to run on a small subset
DEBUG_SIZE = 100  # Number of samples to use in debug mode

# Training Parameters
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3  # Initial learning rate
WEIGHT_DECAY = 1e-4  # Weight decay for optimizer

# Model Architecture
MODEL_NAME = "efficientnet_b0"
EMBEDDING_DIM = 512

# ArcFace Parameters
MARGIN = 0.50
SCALE = 30.0


# -----------------------------------------------------------------------------
# Dynamic Configuration
# -----------------------------------------------------------------------------
def get_num_classes(metadata_path=TRAIN_METADATA_PATH):
    """
    Determines the number of unique whale IDs in the training set,
    excluding the 'new_whale' class.
    """
    if not os.path.exists(metadata_path):
        # Fallback for initialization if metadata doesn't exist yet
        return 4028

    try:
        df = pd.read_csv(metadata_path)
        # We only care about known identities for the ArcFace classifier
        unique_ids = df["Id"].unique()
        known_ids = [uid for uid in unique_ids if uid != "new_whale"]
        return len(known_ids)
    except Exception as e:
        print(f"Warning: Could not calculate NUM_CLASSES dynamically: {e}")
        return 4028


NUM_CLASSES = get_num_classes()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def seed_everything(seed=SEED):
    """
    Seeds all random number generators for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Apply seed immediately upon import
seed_everything(SEED)
