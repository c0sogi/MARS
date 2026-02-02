import os
import torch
import numpy as np
import random

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_14"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission File
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL HYPERPARAMETERS
# =============================================================================
# Reproducibility
SEED = 42

# Debugging
DEBUG = False  # Set to True to run on a small subset of data
DEBUG_SAMPLE_SIZE = 100

# Training Strategy
N_FOLDS = 5  # 5-Fold Cross-Validation
NUM_EPOCHS = 50  # Max epochs
PATIENCE = 10  # Early stopping patience
BATCH_SIZE = 32  # Smaller batch size for better generalization on small data

# Optimization (Adam)
LEARNING_RATE = 1e-3  # Constant learning rate
WEIGHT_DECAY = 1e-4  # L2 Regularization

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of DataLoader workers

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
IMG_HEIGHT = 75
IMG_WIDTH = 75
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH)
IN_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Avg ((HH+HV)/2)
NUM_CLASSES = 1  # Binary classification (Ship vs Iceberg)

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================
# Hybrid Wide-SE-ResNet parameters
MODEL_PARAMS = {
    "input_channels": IN_CHANNELS,
    "stem_channels": 64,
    "block_channels": [64, 128, 128, 128],  # 4 Stages with early expansion to 128
    "se_reduction": 16,  # Squeeze-and-Excitation ratio
    "dropout_rate": 0.2,  # Dropout after activation in head
    "use_angle": True,  # Fuse incidence angle
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
