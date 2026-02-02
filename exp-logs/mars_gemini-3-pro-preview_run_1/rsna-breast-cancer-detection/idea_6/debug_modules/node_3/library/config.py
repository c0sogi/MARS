import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_6"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Files
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = "submission.csv"  # Final submission in root or as required

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Input Dimensions
IMG_HEIGHT = 768
IMG_WIDTH = 768
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Channels: 3 (Image + Spatially Broadcasted Age + Spatially Broadcasted Implant)
IN_CHANNELS = 3

# Normalization (Standard Scaling for Age - approximate values from analysis)
AGE_MEAN = 58.68
AGE_STD = 10.04

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
BACKBONE = "efficientnet_b2"
PRETRAINED = True
DROP_RATE = 0.3
DROP_PATH_RATE = 0.2

# Siamese Configuration
USE_SIAMESE = True
FUSION_METHOD = (
    "spatial_difference_concat"  # Options: global_diff, spatial_difference_concat
)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 8  # Adjusted for A100 memory with 768x768 paired images + gradients
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Standard for AdamW

# Loss Function Weights
# Imbalance Ratio is approx 1:47. Using aggressive positive weighting.
POS_WEIGHT = 47.0

# Gradient Handling
MAX_GRAD_NORM = None  # Gradient clipping disabled as per strategy
ACCUMULATION_STEPS = 1

# =============================================================================
# HARDWARE & PERFORMANCE
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # 12 vCPUs available, 4 is usually a safe sweet spot for dataloading
PIN_MEMORY = False

# =============================================================================
# DEBUGGING & DEVELOPMENT
# =============================================================================
# Set to True to run on a small subset of data for pipeline verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 200
