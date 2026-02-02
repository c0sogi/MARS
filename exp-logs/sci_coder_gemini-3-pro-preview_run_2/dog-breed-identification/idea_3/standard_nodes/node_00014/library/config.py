import os
import torch
from torchvision.models import (
    convnext_large,
    ConvNeXt_Large_Weights,
    swin_v2_b,
    Swin_V2_B_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
)

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_3")

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# ==========================================
# System Configuration
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)
BATCH_SIZE = 32

# ==========================================
# Model Specifications
# ==========================================
# We define a dictionary to hold model constructors, weights, and feature dimensions.
# This allows for easy iteration and strict weight provenance using torchvision.

MODELS = {
    "convnext_large": {
        "constructor": convnext_large,
        "weights": ConvNeXt_Large_Weights.IMAGENET1K_V1,
        "embedding_dim": 1536,  # ConvNeXt Large last channel dimension
        "cache_prefix": "convnext",
    },
    "efficientnet_v2_l": {
        "constructor": efficientnet_v2_l,
        "weights": EfficientNet_V2_L_Weights.IMAGENET1K_V1,
        "embedding_dim": 1280,
        "cache_prefix": "efficientnet_v2_l",
    },
}
