import os
import torch

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
SEED = 42
DEBUG = False  # Set to True to run on a small subset for debugging
MAX_DEBUG_SAMPLES = 100  # Number of samples to use when DEBUG is True

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_39"
SUBMISSION_DIR = "./submission"

# Ensure working and output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Raw Data (JSON)
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata (CSV)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Caching (for processed numpy arrays)
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Checkpoints
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Final Submission
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA HYPERPARAMETERS
# =============================================================================
IMAGE_SIZE = 75
NUM_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Average ((HH+HV)/2)
NUM_CLASSES = 1  # Binary classification (Iceberg vs Ship)

# =============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# =============================================================================
# Convolutional Backbone
LEAKY_RELU_SLOPE = 0.1
# DropBlock Regularization
DROPBLOCK_BLOCK_SIZE = 5  # Size of the block to drop
DROPBLOCK_MAX_PROB = 0.1  # Maximum probability of dropping a block (linear schedule)
DROPBLOCK_START_PROB = 0.0  # Starting probability
# Classification Head
DROPOUT_RATE = 0.5  # Dropout rate in the final linear head

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
N_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 75
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # L2 Regularization
PATIENCE = 12  # Early stopping patience

# =============================================================================
# COMPUTE RESOURCES
# =============================================================================
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
