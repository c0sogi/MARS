import os
import random
import numpy as np
import torch

# ==========================================
# Reproducibility
# ==========================================
SEED = 42


def set_seed(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize seeds immediately upon import
set_seed(SEED)

# ==========================================
# Hardware & Precision Configuration
# ==========================================
# The strategy requires float64 for numerical stability in the OAS estimator
# and the linear inference kernel.
FLOAT_PRECISION = torch.float64
NP_FLOAT_PRECISION = np.float64

# Detect GPU availability
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ==========================================
# Path Configuration
# ==========================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific Metadata Paths (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache Directory for Idea 70
# Used for storing intermediate processed features (parquet/npy)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_70")

# Submission Output
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Image Directory
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# ==========================================
# Directory Initialization
# ==========================================
# Ensure necessary writeable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Processing Constants
# ==========================================
# Target column name in the metadata/dataset
TARGET_COL = "species"

# Image Processing Constants
# These might be used if image resizing is required, though the strategy
# focuses on contour extraction from original binary images.
IMG_SIZE = (224, 224)
