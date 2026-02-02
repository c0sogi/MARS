import os

# =============================================================================
# Global Configuration
# =============================================================================

# Random Seed for Reproducibility
SEED = 42

# =============================================================================
# File Paths and Directories
# =============================================================================

# Input Data
INPUT_DIR = "./input"
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata (Pre-generated)
METADATA_DIR = "./metadata"
TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
VAL_META = os.path.join(METADATA_DIR, "val.csv")
TEST_META = os.path.join(METADATA_DIR, "test.csv")

# Working Directories for Idea 67
WORKING_DIR = "./working/idea_67"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

# Submission Output
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# Data Parameters
# =============================================================================

IMAGE_SIZE = 75
INPUT_CHANNELS = 3  # HH, HV, Average((HH+HV)/2)

# =============================================================================
# Training Hyperparameters
# =============================================================================

NUM_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 75
LEARNING_RATE = 1e-3
PATIENCE = 12
WEIGHT_DECAY = 1e-4  # L2 Regularization

# Debugging / Development
DEBUG = False  # Set to True to limit dataset size for quick testing
MAX_DEBUG_SAMPLES = 100

# =============================================================================
# Model Architecture Hyperparameters
# =============================================================================

# Backbone: Plain CNN
# Channel expansion strategy: 64 -> 128 -> 128 -> 128
BACKBONE_CHANNELS = [64, 128, 128, 128]

# Leaky Squeeze-and-Excitation
SE_REDUCTION = 16
LEAKY_RELU_SLOPE = 0.1

# Isomorphic Readout
# Projection dimension for Stage 3 and Stage 4 features
READOUT_PROJ_DIM = 64

# Classification Head
# Input dim = (64_max3 + 64_min3 + 64_max4 + 64_min4) + 1_angle = 257
CLASSIFIER_HIDDEN_DIM = 256
DROPOUT_RATE = 0.5
