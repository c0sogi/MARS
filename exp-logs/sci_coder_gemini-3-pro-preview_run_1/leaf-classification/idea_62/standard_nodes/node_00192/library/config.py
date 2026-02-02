import os

# ==========================================
# Global Configuration
# ==========================================
# Fixed random seed for reproducibility across libraries
SEED = 42

# Precision setting for the pipeline (High-Precision OAS)
FLOAT_PRECISION = "float64"

# ==========================================
# Directory Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Cache Directory for this specific idea iteration
# Stores intermediate parquet/npy files to speed up subsequent runs
CACHE_DIR = os.path.join(WORKING_DIR, "idea_63")

# ==========================================
# File Paths
# ==========================================
# Metadata CSVs (Generated via Stratified Split)
TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Images Directory (Base path for image loading)
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Sample Submission File (For header/format reference)
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Final Submission Output Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Feature Configuration
# ==========================================
# The 6 specific scalar geometric descriptors to be extracted and fused.
# These names must match the keys used in the dictionary assembly.
GEOMETRIC_FEATURES = [
    "Area",
    "Eccentricity",
    "Solidity",
    "Extent",
    "Aspect_Ratio",
]

# Prefixes for the provided pre-extracted tabular features (192 total)
TABULAR_FEATURE_PREFIXES = ["margin", "shape", "texture"]

# ==========================================
# Hyperparameters & Runtime Controls
# ==========================================
# Controls for dataset size to allow for fast debugging cycles
DEBUG_MODE = False
DEBUG_SAMPLE_SIZE = 100  # Only used if DEBUG_MODE is True

# Model specific constants
NUM_CLASSES = 99
VARIANCE_THRESHOLD = 0.0  # Threshold for removing constant features


# ==========================================
# Setup Logic
# ==========================================
def setup_directories():
    """
    Ensures that the necessary working, cache, and submission directories exist.
    This is called automatically upon module import.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


# Execute setup immediately
setup_directories()
