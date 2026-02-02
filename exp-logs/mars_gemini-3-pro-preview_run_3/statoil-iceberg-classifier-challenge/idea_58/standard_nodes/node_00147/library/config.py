import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Experiment specific directory
IDEA_ID = "idea_58"
IDEA_DIR = os.path.join(WORKING_DIR, IDEA_ID)

# Sub-directories for artifacts
CACHE_DIR = IDEA_DIR
CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(IDEA_DIR, "submission")

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Data File Paths
TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
IMAGE_HEIGHT = 75
IMAGE_WIDTH = 75
# Input Channels: 3 -> Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
INPUT_CHANNELS = 3
NUM_CLASSES = 1  # Binary: 0=Ship, 1=Iceberg

# =============================================================================
# MODEL HYPERPARAMETERS (CDI-CNN)
# =============================================================================
# Backbone: Plain CNN with 4 sequential blocks
# Width Strategy: Expand early, then cap width
BACKBONE_CHANNELS = [64, 128, 128, 128]

# Attention: Standard Squeeze-and-Excitation
SE_REDUCTION_RATIO = 16

# Activation: LeakyReLU to preserve negative signal values
LEAKY_RELU_SLOPE = 0.1

# Regularization
DROPOUT_RATE = 0.5
USE_BIAS = True  # Explicitly retain bias in Conv layers

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
SEED = 42
NUM_FOLDS = 5
BATCH_SIZE = 32

# Optimization Strategy
# Using AdamW with a constant learning rate (no scheduler)
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 Regularization

# Training Duration
NUM_EPOCHS = 75
PATIENCE = 12  # Early stopping patience

# Debugging
# Set to an integer (e.g., 100) to limit dataset size for fast debugging
# Set to None for full training
DEBUG_SAMPLE_SIZE = None

# =============================================================================
# COMPUTE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 12 vCPUs available, leaving some overhead
NUM_WORKERS = 4
