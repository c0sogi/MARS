import os
import torch

# =============================================================================
# FILE PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"

# Ensure the working directory exists for caching and outputs
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Submission output path
SUBMISSION_PATH = "submission.csv"

# =============================================================================
# DATA PREPROCESSING HYPERPARAMETERS
# =============================================================================
# Statistical Projection (StatProj) settings
# We use the central Z-slice range (22-42) to compute Max, Mean, and Std projections.
# This range captures the ink signal while minimizing depth-wise noise.
Z_START = 22
Z_END = 42  # Exclusive upper bound for python slicing (i.e., range is 22 to 41)
# Note: If inclusive behavior is required by the loader, this can be adjusted,
# but standard python slice [22:42] gives 20 slices centered roughly on slice 32.

# Input dimensions
TILE_SIZE = 512
STRIDE = 512  # Stride for tiling (non-overlapping for training, can be adjusted for inference)

# Normalization statistics derived from the training set (Slice 32)
# These are used to normalize the 16-bit input data to a standardized range.
PIXEL_MEAN = 24059.5328
PIXEL_STD = 8677.4554

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Architecture: ResNet18 U-Net
ENCODER_NAME = "resnet18"
ENCODER_WEIGHTS = "imagenet"

# Input configuration
# 1 Channel:
#   1. Maximum Intensity Projection (MIP)
IN_CHANNELS = 1

# Output configuration
CLASSES = 1
ACTIVATION = None  # We use BCEWithLogitsLoss, so no activation at the output layer

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 16  # Increased for ResNet18 and 1-channel input
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

# Optimization & Validation
PATIENCE = 5  # Patience for learning rate scheduler or early stopping
THRESHOLD = 0.5  # Initial threshold for binarization (optimized during validation)
BASELINE_SCORE = 0.474  # Minimum F0.5 score required to save a submission

# Debugging / Development
# Limit the number of training samples for quick debugging runs. Set to None for full training.
MAX_TRAIN_SAMPLES = None
