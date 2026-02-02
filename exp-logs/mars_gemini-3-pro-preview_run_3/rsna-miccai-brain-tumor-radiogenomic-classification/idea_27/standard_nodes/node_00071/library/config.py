import os
import torch

# ==========================================
# Global Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_27"
SUBMISSION_DIR = "./submission"

# Metadata Paths (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Output Paths
# Ensure directories exist immediately upon import
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
IMG_SIZE = 224
NUM_SLICES_PER_MODALITY = 32
NUM_MODALITIES = 4
# Total input channels = 32 slices * 4 modalities = 128
TOTAL_INPUT_CHANNELS = NUM_SLICES_PER_MODALITY * NUM_MODALITIES

# Sorting and Sampling
# We use Instance Number for sorting, not filenames
USE_INSTANCE_NUM_SORTING = True

# ==========================================
# Model Configuration
# ==========================================
# Backbone settings
BACKBONE_NAME = "efficientnet_b0"
# The stem compresses 128 channels -> 64 channels
STEM_OUT_CHANNELS = 64
# Stochastic depth rate for regularization
DROP_PATH_RATE = 0.2

# ==========================================
# Training Configuration
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BATCH_SIZE = 16  # Adjusted for A100 40GB with high-channel input
EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 4

# Optimization
USE_WEIGHT_DECAY = False  # Adam without weight decay as per idea
PATIENCE = 5  # For early stopping

# ==========================================
# Debugging & Development
# ==========================================
# Set to an integer (e.g., 50) to train/predict on a subset of data.
# Set to None for full run.
DEBUG_SAMPLE_SIZE = None
