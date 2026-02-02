import os
import torch

# ==========================================
#              PATHS & DIRECTORIES
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_4"
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# ==========================================
#              DATA CONFIGURATION
# ==========================================
IMAGE_SIZE = 32
NUM_CLASSES = 1  # Binary classification
NUM_WORKERS = 2  # Number of data loading workers

# Debugging / Development
# Set DEBUG to True to run on a small subset of data for quick pipeline testing
DEBUG = False
DEBUG_SAMPLE_SIZE = 500  # Number of samples to use in debug mode

# ==========================================
#          MODEL CONFIGURATION
# ==========================================
# Custom Micro-ConvNeXt Architecture settings
# Designed for 32x32 inputs to maintain 8x8 resolution at the final stage
# Structure: Stem (Stride 1) -> Stage 1 -> Downsample -> Stage 2 -> Downsample -> Stage 3
MODEL_DIMS = [64, 128, 256]  # Channel dimensions for each stage
MODEL_DEPTHS = [3, 3, 3]  # Number of blocks per stage
KERNEL_SIZE = 7  # Kernel size for depthwise convolutions
DROP_PATH_RATE = 0.1  # Stochastic depth rate for regularization
LAYER_SCALE_INIT_VALUE = 1e-6  # Initial value for layer scale

# ==========================================
#          TRAINING CONFIGURATION
# ==========================================
# Homogeneous Seed Averaging Strategy
SEEDS = [42, 0, 1, 2, 3]

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05  # AdamW weight decay
PATIENCE = 5  # Early stopping patience (epochs without improvement)

# Augmentation
USE_LIGHT_AUGMENTATION = True  # Enforce Horizontal/Vertical flips only

# ==========================================
#              COMPUTE
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
