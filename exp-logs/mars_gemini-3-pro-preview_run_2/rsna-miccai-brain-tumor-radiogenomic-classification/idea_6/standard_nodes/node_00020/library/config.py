import os
import torch

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
SEED = 42

# -----------------------------------------------------------------------------
# Directories and File Paths
# -----------------------------------------------------------------------------
# Input Data
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata (Pre-generated)
METADATA_DIR = "./metadata"
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

# Working Directory (for artifacts)
WORKING_DIR = "./working/idea_7"
os.makedirs(WORKING_DIR, exist_ok=True)

# Output Paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
ROI_CACHE_PATH = os.path.join(WORKING_DIR, "roi_cache.parquet")
SUBMISSION_PATH = "submission.csv"

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# Architecture
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 1
IMG_SIZE = 224

# Input Dimensions
# 4 Modalities (FLAIR, T1w, T1wCE, T2w) * 3 Slices each = 12 Channels
NUM_MODALITIES = 4
SLICES_PER_MODALITY = 3
IN_CHANNELS = NUM_MODALITIES * SLICES_PER_MODALITY
STEM_GROUPS = 4  # Grouped convolution in the first layer

# Training
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive regularization to prevent overfitting
PATIENCE = 5  # Early stopping patience

# -----------------------------------------------------------------------------
# Data Processing & ROI Selection
# -----------------------------------------------------------------------------
# ROI Selection Heuristics
# Exclude top/bottom 15% of the volume to remove neck/vertex artifacts
EXCLUDE_TOP_BOTTOM_RATIO = 0.15
# Stride for 2.5D stacking (e.g., Peak-5, Peak, Peak+5)
STRIDE = 5

# -----------------------------------------------------------------------------
# Hardware Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Adjust based on CPU core count
