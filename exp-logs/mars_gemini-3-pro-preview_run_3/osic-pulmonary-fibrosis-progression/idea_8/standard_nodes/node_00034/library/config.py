import os

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata Files
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Image Directory
# Note: Metadata contains relative paths (e.g., 'train/ID...'), so we point to input root
IMAGE_DIR = INPUT_DIR

# Output Directories
WORKING_DIR = "./working/idea_8"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Statistics (from EDA)
# ==========================================
# These are used for Z-score normalization of inputs and targets
STATS = {
    "FVC_MEAN": 2654.6528,
    "FVC_STD": 801.7017,
    "WEEKS_MEAN": 31.3751,
    "WEEKS_STD": 23.4602,
    "PERCENT_MEAN": 76.9105,
    "PERCENT_STD": 19.1970,
    "AGE_MEAN": 67.5825,
    "AGE_STD": 6.6259,
}

# ==========================================
# Model Hyperparameters
# ==========================================
SEED = 42

# Architecture
IMG_SIZE = 256
NUM_SLICES = 3  # Apical, Middle, Basal selection strategy
EMBEDDING_DIM = 128  # Dimension for image feature projection

# Training
BATCH_SIZE = 32
EPOCHS = 50
NUM_WORKERS = 4

# Optimization
LR_BACKBONE = 1e-4  # Lower learning rate for pre-trained EfficientNet
LR_HEAD = 1e-3  # Higher learning rate for MLP head
WEIGHT_DECAY = 1e-2
PATIENCE = 10  # Early stopping patience

# Loss Weights
LAMBDA_MSE = 1.0  # Weight for auxiliary slope supervision

# ==========================================
# Metric & Post-Processing
# ==========================================
MIN_UNCERTAINTY = 70.0  # Clipped sigma for metric calculation
UNCERTAINTY_FLOOR = 0.05  # Epsilon for numerical stability in NLL loss
MAX_ERROR = 1000.0  # Max absolute error for metric calculation

# ==========================================
# Debugging
# ==========================================
DEBUG = False
DEBUG_SIZE = 20  # Number of patients to use when DEBUG is True
