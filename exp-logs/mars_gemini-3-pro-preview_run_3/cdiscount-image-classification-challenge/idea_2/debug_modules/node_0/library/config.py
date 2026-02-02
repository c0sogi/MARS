import os
import torch
import numpy as np
import random

# ==========================================
# DIRECTORY CONFIGURATION
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# FILE PATHS
# ==========================================
TRAIN_BSON_PATH = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON_PATH = os.path.join(INPUT_DIR, "test.bson")

TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

CATEGORY_NAMES_PATH = os.path.join(INPUT_DIR, "category_names.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output paths
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
MODEL_NAME = "resnet18"
IMG_SIZE = 224
NUM_CLASSES = 5270
EMBEDDING_DIM = 512  # ResNet18 fc layer input size

# ==========================================
# TRAINING & OPTIMIZATION
# ==========================================
# Batch size for the Feature Extraction phase (Images -> CNN -> Embeddings)
# Limited by GPU memory (A100 40GB can handle large batches, but safe default)
FE_BATCH_SIZE = 256

# Batch size for the MLP Training phase (Embeddings -> MLP -> Class)
# Input is small (512 floats), so we can use very large batches
MLP_BATCH_SIZE = 4096

LEARNING_RATE = 1e-3
EPOCHS = 20
PATIENCE = 5  # Early stopping patience

# Hardware settings
NUM_WORKERS = 12  # Utilizing available vCPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# REPRODUCIBILITY & UTILS
# ==========================================
SEED = 42


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Debugging / Development
# Set to an integer (e.g., 10000) to limit dataset size for fast prototyping
# Set to None for full run
DEBUG_SAMPLE_SIZE = None
