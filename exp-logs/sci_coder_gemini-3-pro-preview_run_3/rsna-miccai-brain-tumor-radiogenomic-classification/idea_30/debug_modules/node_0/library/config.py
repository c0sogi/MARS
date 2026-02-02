import os
import torch

# ==========================================
# Path Configurations
# ==========================================
# Input Directories
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata Files
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output & Working Directories
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_30")
CACHE_DIR = IDEA_DIR  # Directory for caching numpy arrays
MODEL_SAVE_PATH = os.path.join(IDEA_DIR, "best_model.pth")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Configurations
# ==========================================
IMG_SIZE = 320
NUM_SLICES_PER_MODALITY = 16
MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
NUM_MODALITIES = len(MODALITIES)
TOTAL_CHANNELS = NUM_SLICES_PER_MODALITY * NUM_MODALITIES  # 16 * 4 = 64

# ==========================================
# Model Configurations
# ==========================================
MODEL_NAME = "efficientnet_b0"
DROP_PATH_RATE = 0.2
NUM_CLASSES = 1

# ==========================================
# Training Configurations
# ==========================================
SEED = 42
BATCH_SIZE = 8  # Adjusted for 320x320x64 input tensor
LEARNING_RATE = 1e-4
EPOCHS = 15
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Utility Functions
# ==========================================
def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
