import os
import torch

# -----------------------------------------------------------------------------
# Directory Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Working directory for caching processed data and saving models (Idea 11)
WORKING_DIR = "./working/idea_11"
SUBMISSION_DIR = "./submission"

# Ensure necessary writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Metadata File Paths
# -----------------------------------------------------------------------------
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# -----------------------------------------------------------------------------
# Global Training Hyperparameters
# -----------------------------------------------------------------------------
# Training duration: Fully converged independent bagging
EPOCHS = 1000

# Optimization parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# Hardware configuration
# Utilizing available 12 vCPUs and NVIDIA A100 GPU
NUM_WORKERS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global seed for deterministic operations outside the ensemble variance
GLOBAL_SEED = 42

# -----------------------------------------------------------------------------
# Ensemble Strategy Configuration
# -----------------------------------------------------------------------------
# The ensemble consists of two streams designed to align bottleneck resolution.

# Stream A: Context Specialist
# Architecture: Deep 4-Level U-Net
# Input: 320x320 patches
# Bottleneck: 320 / 16 = 20x20 spatial resolution
PATCH_SIZE_CONTEXT = 320
CONTEXT_MODEL_CONFIG = {
    "type": "context",
    "patch_size": PATCH_SIZE_CONTEXT,
    "unet_depth": 4,
    "encoder_filters": [32, 64, 128, 256, 512],
}
# Seeds for the 5 independent Context models
SEEDS_CONTEXT = [42, 43, 44, 45, 46]

# Stream B: Diversity Specialist
# Architecture: Lightweight 3-Level U-Net
# Input: 160x160 patches
# Bottleneck: 160 / 8 = 20x20 spatial resolution (Aligned with Stream A)
PATCH_SIZE_DIVERSITY = 160
DIVERSITY_MODEL_CONFIG = {
    "type": "diversity",
    "patch_size": PATCH_SIZE_DIVERSITY,
    "unet_depth": 3,
    "encoder_filters": [32, 64, 128, 256],
}
# Seeds for the 5 independent Diversity models
SEEDS_DIVERSITY = [47, 48, 49, 50, 51]

# -----------------------------------------------------------------------------
# Inference Configuration
# -----------------------------------------------------------------------------
# D4 Group Test-Time Augmentation (8 views: 4 Rotations x 2 Flips)
TTA_VIEWS = 8
