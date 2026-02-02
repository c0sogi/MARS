import os
import torch

# ====================================================
# Directory & Path Configurations
# ====================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Image Directory Root
# Note: Metadata 'file_path' columns are relative to INPUT_DIR (e.g., "jpeg/train/ISIC_...jpg")
IMAGE_DIR = INPUT_DIR

# ====================================================
# Data Configuration
# ====================================================
IMG_SIZE = 384
NUM_CLASSES = 1
NUM_WORKERS = 8  # Optimized for 12 vCPUs

# Tabular Feature Definitions
# These match the columns identified in the data analysis
NUM_COLS = ["age_approx"]
CAT_COLS = ["sex", "anatom_site_general_challenge"]

# ====================================================
# Model Configuration
# ====================================================
MODEL_NAME = "efficientnet_b3"
PRETRAINED = True
DROP_RATE = 0.3  # Dropout rate for the classifier head
DROP_PATH_RATE = 0.2  # Stochastic depth rate

# Tabular Model Settings
TAB_EMBED_DIM = 16
TAB_HIDDEN_DIM = 64

# ====================================================
# Training Hyperparameters
# ====================================================
SEED = 42
EPOCHS = 10
BATCH_SIZE = 32  # Conservative size for 384x384 on A100 to ensure stability
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01  # Standard weight decay as per strategy
POS_WEIGHT = (
    15.0  # Dampened positive weight to handle class imbalance (approx 1:55 ratio)
)
MAX_GRAD_NORM = 10.0
PATIENCE = 3  # Early stopping patience

# ====================================================
# Compute Configuration
# ====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================================================
# Debugging & Development
# ====================================================
# Set DEBUG to True to run on a small subset of data for pipeline verification
DEBUG = False
MAX_DEBUG_SAMPLES = 500
