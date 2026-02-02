import os

# =============================================================================
# DIRECTORY CONFIGURATION
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_29"

# Specific output directories
# We use os.makedirs in the respective modules, but define paths here
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = WORKING_DIR  # Submission file goes to the idea root

# File paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
SEED = 42
DEBUG = False  # Set to True to run on a small subset for debugging
DEBUG_SUBSET_SIZE = 100

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
# Input image dimensions
IMG_HEIGHT = 75
IMG_WIDTH = 75
# 3 Channels: HH, HV, and Average((HH+HV)/2)
IN_CHANNELS = 3

# Data Loading
NUM_WORKERS = 2  # Adjust based on vCPU availability (12 vCPUs available)
PIN_MEMORY = True

# =============================================================================
# MODEL ARCHITECTURE CONFIGURATION
# =============================================================================
# Efficient-Attentive Plain CNN (EAP-CNN) settings
# Sequential channel expansion: 64 -> 128 -> 128 -> 128
MODEL_CHANNELS = [64, 128, 128, 128]

# Activation settings
LEAKY_RELU_SLOPE = 0.1

# Regularization
DROPOUT_RATE = 0.5  # Applied after the linear layer activation

# Attention Mechanism
ECA_KERNEL_SIZE = 3  # Kernel size for 1D convolution in ECA module

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Cross-Validation
NUM_FOLDS = 5

# Optimization
BATCH_SIZE = 32
NUM_EPOCHS = 75
PATIENCE = 12  # Early stopping patience

# Optimizer settings (Adam)
LEARNING_RATE = 1e-3  # Constant learning rate
WEIGHT_DECAY = 1e-4  # L2 Regularization

# Inference
USE_TTA = False  # Test-Time Augmentation explicitly disabled
