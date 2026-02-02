import os
import torch
import random
import numpy as np

# ==========================================
# Reproducibility
# ==========================================
SEED = 42


def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# System & Compute
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# ==========================================
# File Paths & Directories
# ==========================================
INPUT_ROOT = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Create output directories if they don't exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

# Output Paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Model Hyperparameters
# ==========================================
MODEL_NAME = "convnext_tiny"  # Architecture to use
PRETRAINED = True  # Initialize with ImageNet weights
NUM_CLASSES = 23  # Total number of categories (0-22)
IMAGE_SIZE = 224  # Input resolution

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 64  # Batch size for A100 GPU
NUM_EPOCHS = 15  # Max training epochs
LEARNING_RATE = 1e-4  # Initial learning rate
WEIGHT_DECAY = 1e-2  # Weight decay for AdamW
FOCAL_GAMMA = 2.0  # Gamma parameter for Focal Loss
EARLY_STOPPING_PATIENCE = 5  # Stop if validation metric doesn't improve

# ==========================================
# Label Mapping
# ==========================================
ID2NAME = {
    0: "empty",
    1: "deer",
    2: "moose",
    3: "squirrel",
    4: "rodent",
    5: "small_mammal",
    6: "elk",
    7: "pronghorn_antelope",
    8: "rabbit",
    9: "bighorn_sheep",
    10: "fox",
    11: "coyote",
    12: "black_bear",
    13: "raccoon",
    14: "skunk",
    15: "wolf",
    16: "bobcat",
    17: "cat",
    18: "dog",
    19: "opossum",
    20: "bison",
    21: "mountain_goat",
    22: "mountain_lion",
}
