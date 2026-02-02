import os
import torch
import numpy as np
import random

# =============================================================================
# 1. Path Configurations
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

# Specific File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
UNICODE_TRANSLATION_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# 2. Data & Model Hyperparameters
# =============================================================================
# Image dimensions
IMG_SIZE = 1024  # Resizing target (Square)

# Dataset specifics
NUM_CLASSES = 3848  # Number of unique classes identified in the training set analysis
MAX_DETECTIONS = 1200  # Submission limit per page

# Model Architecture
BACKBONE = "resnet18"
NECK_CHANNELS = (
    256  # Channels in the neck feature map (Cite solution_lesson_node_00001)
)
HEAD_CHANNELS = (
    512  # Dimensionality of the embedding head (Cite solution_lesson_node_00001)
)

# Inference
CONF_THRESHOLD = 0.1  # Minimum confidence score to consider a detection

# =============================================================================
# 3. Training Hyperparameters
# =============================================================================
SEED = 42
BATCH_SIZE = 16  # Suitable for A100-40GB with 1024x1024 images
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_WORKERS = 4  # For DataLoader
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Debugging flag to use a smaller dataset subset
DEBUG = False
DEBUG_SAMPLE_SIZE = 100


# =============================================================================
# 4. Utility Functions
# =============================================================================
def setup_directories():
    """
    Creates the necessary directories for working files, cache, and submissions.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
