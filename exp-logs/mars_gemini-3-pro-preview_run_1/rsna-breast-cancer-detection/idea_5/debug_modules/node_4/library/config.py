import os
import torch
import random
import numpy as np

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
# Input Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Working Directory (Idea 5 specific)
WORKING_DIR = "./working/idea_5"
os.makedirs(WORKING_DIR, exist_ok=True)

# Output Paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = "./submission/submission.csv"
# Ensure submission directory exists
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# Cache Directory for Processed Data
CACHE_DIR = WORKING_DIR

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
# Model Architecture
MODEL_NAME = "efficientnet_b2"
IMAGE_SIZE = 768
IN_CHANNELS = 3  # Image (1) + Age (1) + Implant (1)
NUM_CLASSES = 1

# Training
BATCH_SIZE = 16  # Adjusted for A100 40GB with 768x768 Siamese Network
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
GRADIENT_CLIPPING = False  # Disabled as per Idea 5 instructions
POS_WEIGHT_VAL = 47.0  # Aggressive weighting for imbalance

# Optimizer & Scheduler
WEIGHT_DECAY = 1e-2
T_MAX = NUM_EPOCHS  # For Cosine Annealing

# =============================================================================
# COMPUTE & REPRODUCIBILITY
# =============================================================================
SEED = 42
NUM_WORKERS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_MIXED_PRECISION = True


def seed_everything(seed: int = SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# Column Names
ID_COL = "prediction_id"
TARGET_COL = "cancer"
PATIENT_ID_COL = "patient_id"
IMAGE_ID_COL = "image_id"
FILE_PATH_COL = "file_path"
LATERALITY_COL = "laterality"
VIEW_COL = "view"
AGE_COL = "age"
IMPLANT_COL = "implant"

# Feature Flags
USE_AGE = True
USE_IMPLANT = True
