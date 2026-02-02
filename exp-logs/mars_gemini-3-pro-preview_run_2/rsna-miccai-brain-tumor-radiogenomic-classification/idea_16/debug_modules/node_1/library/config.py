import os
import torch
import random
import numpy as np

# -----------------------------------------------------------------------------
# Paths & Directories
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Files
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
CACHE_FILE_PATH = os.path.join(WORKING_DIR, "roi_cache.parquet")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Hyperparameters
# -----------------------------------------------------------------------------
IMG_SIZE = 224
NUM_SLICES = 3  # 3 slices per modality: Anchor-5, Anchor, Anchor+5
STRIDE = 5  # Fixed stride for neighbor selection
DEPTH_MIN = 0.15  # Ignore top 15% of volume (skull/scalp)
DEPTH_MAX = 0.85  # Ignore bottom 15% of volume (neck/jaw)

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
MODEL_NAME = "efficientnet_b0"
MODALITY_DROPOUT_PROB = 0.2  # Probability to drop an entire modality group
HEAD_DROPOUT_PROB = 0.5  # Dropout in the classification head
NUM_CLASSES = 1

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  # Low LR to preserve pre-trained features
WEIGHT_DECAY = 1e-2  # Aggressive weight decay for regularization
NUM_EPOCHS = 20
NUM_WORKERS = 4  # Multi-threaded data loading
SEED = 42

# -----------------------------------------------------------------------------
# Compute Configuration
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
