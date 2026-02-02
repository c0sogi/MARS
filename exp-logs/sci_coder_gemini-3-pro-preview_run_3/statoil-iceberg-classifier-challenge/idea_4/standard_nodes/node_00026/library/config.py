import os
import random
import numpy as np
import torch

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Files
TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

# Cached Data Files (Numpy format for speed)
# We use version suffix _v6 to match the idea_6 folder structure
TRAIN_IMAGES_FILE = os.path.join(WORKING_DIR, "X_train_v6.npy")
TRAIN_ANGLES_FILE = os.path.join(WORKING_DIR, "angle_train_v6.npy")
TRAIN_LABELS_FILE = os.path.join(WORKING_DIR, "y_train_v6.npy")

TEST_IMAGES_FILE = os.path.join(WORKING_DIR, "X_test_v6.npy")
TEST_ANGLES_FILE = os.path.join(WORKING_DIR, "angle_test_v6.npy")
TEST_IDS_FILE = os.path.join(WORKING_DIR, "test_ids_v6.npy")

# Output Files
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA PARAMETERS
# =============================================================================
IMAGE_SIZE = 75
NUM_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Average (HH+HV)/2
NUM_CLASSES = 1  # Binary classification (0: Ship, 1: Iceberg)

# Normalization constants (if needed globally, though often computed per batch/dataset)
# These are approximate values derived from data analysis
BAND_1_MEAN = -20.5754
BAND_1_STD = 5.2486
BAND_2_MEAN = -26.2593
BAND_2_STD = 3.3965

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# SE-CNN Architecture
SE_REDUCTION_RATIO = 16
CONV_FILTERS = [64, 128, 256, 512]  # Filters for the 4 conv blocks
DENSE_UNITS = 512
DROPOUT_RATE = 0.2

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
N_FOLDS = 5
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4  # Constant learning rate
WEIGHT_DECAY = 0.0  # Adam default
PATIENCE = 10  # Early stopping patience

# =============================================================================
# COMPUTE AND REPRODUCIBILITY
# =============================================================================
SEED = 42
NUM_WORKERS = 4  # Number of subprocesses for data loading
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# DEBUGGING
# =============================================================================
DEBUG = False  # Set to True to run on a small subset
DEBUG_SIZE = 100  # Number of samples to use in debug mode


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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
