import os
import torch

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

# Metadata paths (pre-generated)
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Working directory for caching processed data and saving models
# Specific to the current experimental idea
WORKING_DIR = "./working/idea_16"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Model checkpoint path
MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# =============================================================================
# DATA HYPERPARAMETERS
# =============================================================================
SEED = 42
IMG_SIZE = 256
NUM_SLICES = 32
NUM_MODALITIES = 4  # FLAIR, T1w, T1wCE, T2w

# The model takes a 2.5D stack.
# Input channels = Slices * Modalities = 32 * 4 = 128
TOTAL_INPUT_CHANNELS = NUM_SLICES * NUM_MODALITIES

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
BACKBONE = "efficientnet_b0"
PRETRAINED = True
STEM_OUT_CHANNELS = 64  # Project 128 channels -> 64 channels before backbone
DROP_PATH_RATE = 0.2  # Stochastic depth for regularization
NUM_CLASSES = 1

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
NUM_WORKERS = 4  # Number of DataLoader workers

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
