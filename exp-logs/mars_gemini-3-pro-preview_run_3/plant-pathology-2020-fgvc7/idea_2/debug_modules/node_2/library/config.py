import os
import random
import numpy as np
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
METADATA_DIR = "./metadata"

# Metadata CSV paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Output directories
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary writable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA HYPERPARAMETERS
# =============================================================================
IMG_SIZE = 224
NUM_CLASSES = 4
CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]
NUM_WORKERS = 4

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Models to be used in the ensemble
MODEL_EFFICIENTNET = "tf_efficientnet_b0_ns"
MODEL_CONVNEXT = "convnext_tiny"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

# Stochastic Weight Averaging (SWA) settings
SWA_START_EPOCH = 10
SWA_LR = 5e-5

# Early Stopping
PATIENCE = 5


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def seed_everything(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Global default seed
SEED = 42
