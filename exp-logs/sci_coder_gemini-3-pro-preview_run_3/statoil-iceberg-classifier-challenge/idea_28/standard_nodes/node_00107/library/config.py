import os
import torch

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================

# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_28"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Cache Paths (for deterministic data processing)
CACHE_PATH_X_TRAIN = os.path.join(WORKING_DIR, "X_train.npy")
CACHE_PATH_Y_TRAIN = os.path.join(WORKING_DIR, "y_train.npy")
CACHE_PATH_ANGLE_TRAIN = os.path.join(WORKING_DIR, "angle_train.npy")

CACHE_PATH_X_VAL = os.path.join(WORKING_DIR, "X_val.npy")
CACHE_PATH_Y_VAL = os.path.join(WORKING_DIR, "y_val.npy")
CACHE_PATH_ANGLE_VAL = os.path.join(WORKING_DIR, "angle_val.npy")

CACHE_PATH_X_TEST = os.path.join(WORKING_DIR, "X_test.npy")
CACHE_PATH_IDS_TEST = os.path.join(WORKING_DIR, "ids_test.npy")
CACHE_PATH_ANGLE_TEST = os.path.join(WORKING_DIR, "angle_test.npy")

# =============================================================================
# GENERAL HYPERPARAMETERS
# =============================================================================

SEED = 42
NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_WORKERS = 4  # Adjusted for 12 vCPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training loop settings
NUM_EPOCHS = 75
LEARNING_RATE = 1e-3
PATIENCE = 12
WEIGHT_DECAY = 1e-4  # L2 Regularization

# =============================================================================
# DATA SPECIFICS
# =============================================================================

IMAGE_SIZE = 75
INPUT_CHANNELS = 3  # HH, HV, Average
NUM_CLASSES = 1  # Binary classification (output logits)

# =============================================================================
# MODEL ARCHITECTURE SPECIFICS
# =============================================================================

# Backbone configuration: Plain CNN with 4 blocks
# Width Strategy: 64 -> 128 -> 128 -> 128
CHANNEL_CONFIG = [64, 128, 128, 128]

# Activation settings
LEAKY_RELU_SLOPE = 0.1

# Regularization
DROPOUT_RATE = 0.5

# Feature Fusion
# The incidence angle is concatenated with pooled features from Stage 3 and Stage 4
# Stage 3 (128) + Stage 4 (128) + Angle (1) = 257
FUSION_INPUT_DIM = 128 + 128 + 1
