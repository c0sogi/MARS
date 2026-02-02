import os

# ==========================================
# Global Configuration
# ==========================================

# Reproducibility
RANDOM_SEED = 42

# ==========================================
# Directory Paths
# ==========================================
# Base paths
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
WORKING_DIR = os.path.join(BASE_DIR, "working")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Specific Data Paths
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Caching Directory for Idea 10 (Multi-Fidelity Ensemble)
# Stores extracted features, PCA models, and intermediate pipelines
CACHE_DIR = os.path.join(WORKING_DIR, "idea_10")

# ==========================================
# Model & Feature Extraction Config
# ==========================================
# Image Parameters
IMG_SIZE = 224
BATCH_SIZE = 32  # Optimized for A100-40GB
NUM_WORKERS = 4

# Pretrained Model Checkpoints
# Stream 1: Global Geometry (ViT)
MODEL_DINOV2 = "facebook/dinov2-large"
# Stream 2: Local Texture (CNN)
MODEL_CONVNEXT = "facebook/convnext-large-224-22k-1k"

# Multi-View Canonical Averaging
# Number of rotations to average (0, 90, 180, 270 degrees)
N_VIEWS = 4

# ==========================================
# Ensemble Architecture Config
# ==========================================
# Stratified K-Fold
N_FOLDS = 10

# Multi-Fidelity Branches
# Cumulative variance thresholds for PCA reduction
# Branch 1: High-Fidelity (Max detail, potential noise)
# Branch 2: Mid-Fidelity (Balanced)
# Branch 3: High-Stability (Max robustness, low dimension)
PCA_THRESHOLDS = [0.99, 0.95, 0.90]

# Tabular Preprocessing
# Output distribution for QuantileTransformer
QUANTILE_OUTPUT_DIST = "normal"

# Prediction Clipping
# Avoid log(0) in metric calculation
PROB_CLIP_EPS = 1e-15


# ==========================================
# Setup Utilities
# ==========================================
def setup_directories():
    """
    Ensures that the necessary working and submission directories exist.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
