import os
import torch
import random
import numpy as np

# -----------------------------------------------------------------------------
# General Configuration
# -----------------------------------------------------------------------------
SEED = 42
DEBUG = False
DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use if DEBUG is True

# -----------------------------------------------------------------------------
# Hardware Settings
# -----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2  # Adjust based on available vCPUs (12 available)
PREFETCH_FACTOR = 2

# -----------------------------------------------------------------------------
# Directory & File Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata CSV Paths (Pre-generated in ./metadata)
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Paths for Extracted Features (Parquet format)
TRAIN_FEATS_CACHE = os.path.join(WORKING_DIR, "train_features.parquet")
VAL_FEATS_CACHE = os.path.join(WORKING_DIR, "val_features.parquet")
TEST_FEATS_CACHE = os.path.join(WORKING_DIR, "test_features.parquet")

# -----------------------------------------------------------------------------
# Data Preprocessing Constants
# -----------------------------------------------------------------------------
IMG_SIZE = 224  # Input size for MobileNetV3
BATCH_SIZE = 64

# Standard ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Tabular Data Column Definitions
ID_COL = "image_name"
TARGET_COL = "target"
NUMERICAL_COLS = ["age_approx"]
CATEGORICAL_COLS = ["sex", "anatom_site_general_challenge"]

# -----------------------------------------------------------------------------
# Model Hyperparameters
# -----------------------------------------------------------------------------
# Backbone
MODEL_NAME = "mobilenet_v3_large"
FEATURE_DIM = 1280  # Output dimension of MobileNetV3 Large (with projection layer)

# Classifier (Logistic Regression)
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 1000
LR_CLASS_WEIGHT = "balanced"  # Handles class imbalance
LR_C = 1.0  # Inverse of regularization strength


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
