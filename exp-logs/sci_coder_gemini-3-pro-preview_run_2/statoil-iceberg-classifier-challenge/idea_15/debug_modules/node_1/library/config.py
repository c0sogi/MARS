import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_15"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Paths
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Artifact Paths
PROCESSED_DATA_PATH = os.path.join(CACHE_DIR, "processed_data.npz")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA HYPERPARAMETERS
# =============================================================================
IMAGE_SIZE = 75
NUM_BANDS = 2
# Input channels: Band 1 (HH), Band 2 (HV), Average (HH+HV)/2
NUM_INPUT_CHANNELS = 3
NUM_CLASSES = 1  # Binary classification (Ship vs Iceberg)

# Data Augmentation Settings
# Rotations: 0, 90, 180, 270 degrees
ROTATION_ANGLES = [0, 90, 180, 270]
USE_HORIZONTAL_FLIP = True
USE_VERTICAL_FLIP = False  # Excluded per solution design

# =============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# =============================================================================
# GDP-Net Specifics
CONTRACTED_FILTERS = 32  # Filters in the final conv block before pooling
GATING_VECTOR_DIM = 64  # Dimension of the incidence angle gating vector
DROPOUT_RATE = 0.2  # Dropout probability in the dense head

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
NUM_FOLDS = 5
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
MAX_EPOCHS = 100  # Upper limit, controlled by Early Stopping
PATIENCE = 10  # Early stopping patience
NUM_WORKERS = 4  # Number of DataLoader workers

# =============================================================================
# DEBUGGING AND DEVELOPMENT
# =============================================================================
# Set DEBUG to True to run on a small subset of data for quick testing
DEBUG = False
MAX_DEBUG_SAMPLES = 100  # Number of samples to use when DEBUG is True
