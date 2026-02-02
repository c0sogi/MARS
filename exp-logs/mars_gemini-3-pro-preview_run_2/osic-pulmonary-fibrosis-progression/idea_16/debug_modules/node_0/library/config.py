import os
import torch

# ==========================================
# Directory & File Paths
# ==========================================
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")

METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

WORKING_DIR = "./working"
# Cache directory for idea_16 specific artifacts (features, models)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_16")

SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Reproducibility & Hardware
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2

# ==========================================
# Image Processing Hyperparameters
# ==========================================
# EfficientNet-B0 typically uses 224x224. We resize to 256 before cropping/processing.
IMG_SIZE = 256
# Number of slices to select per patient based on highest pixel variance
N_SLICES = 5
# Batch size for the feature extractor (CNN)
BATCH_SIZE = 32

# ==========================================
# Feature Engineering Hyperparameters
# ==========================================
# Target number of dimensions for PCA reduction of the weighted image embedding
PCA_COMPONENTS = 30

# ==========================================
# Model Architecture & Training Hyperparameters
# ==========================================
# Number of bootstrap models in the bagging ensemble
N_BAGS = 50

# Quantile Regression Settings (FVC Prediction)
# We target the median (0.5) to minimize L1 loss
QUANTILES = [0.5]

# ElasticNet Settings (Uncertainty Prediction)
# L1_RATIO: 1.0 = Lasso, 0.0 = Ridge. 0.5 is a balanced mix.
ELASTIC_L1_RATIO = 0.5
# Regularization strength (alpha) - can be tuned, but fixed here for stability
ELASTIC_ALPHA = 0.1

# ==========================================
# Metric & Post-Processing Constants
# ==========================================
# Laplace Log Likelihood constants
MIN_CONFIDENCE = 70
MAX_ERROR = 1000


def setup_directories():
    """
    Ensures that the necessary working and submission directories exist.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
