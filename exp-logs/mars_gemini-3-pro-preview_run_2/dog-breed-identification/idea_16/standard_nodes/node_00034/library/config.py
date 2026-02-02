import os
import torch
import random
import numpy as np
from torchvision.transforms import InterpolationMode

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
SEED = 42
NUM_CLASSES = 120

# Compute Resources
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Using 8 workers to efficiently utilize the 12 vCPUs without overhead
NUM_WORKERS = 8
# A100 40GB allows for large batch sizes during inference/feature extraction
BATCH_SIZE = 64

# -----------------------------------------------------------------------------
# File System Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for caching intermediate features and models (Idea 16)
WORKING_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"

# Metadata Files
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Model Architecture & Feature Extraction
# -----------------------------------------------------------------------------
MODEL_NAME = "convnext_large"
WEIGHTS = "IMAGENET1K_V1"

# Feature Extraction Nodes (Torchvision ConvNeXt naming convention)
# 'features.5': Stage 3 (Texture/Mid-level) - Requires Normalization
# 'features.7': Stage 4 (Semantic/High-level) - Natively Normalized
FEATURE_NODES = {
    "features.5": "stage3",
    "features.7": "stage4",
}

# -----------------------------------------------------------------------------
# Multi-View Data Pipeline Configuration
# -----------------------------------------------------------------------------
# Explicitly use Bicubic interpolation for all resizing operations
INTERPOLATION = InterpolationMode.BICUBIC

# View Definitions:
# 1. Global: Squish to 224x224 (Preserves topology, distorts aspect ratio)
# 2. Standard: Resize shortest edge to 232, Center Crop 224 (Standard ImageNet)
# 3. Local: Resize shortest edge to 288, Center Crop 224 (Zoomed in for texture)
VIEWS = {
    "global": {"resize": (224, 224), "crop": None},  # Tuple implies fixed size resize
    "standard": {"resize": 232, "crop": 224},  # Integer implies resize shortest edge
    "local": {"resize": 288, "crop": 224},  # Integer implies resize shortest edge
}


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def seed_everything(seed=SEED):
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
