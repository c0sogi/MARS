import os
import torch

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata CSV Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Submission Path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Compute and Environment
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Loading
# A100 40GB can handle larger batches for inference/embedding extraction
BATCH_SIZE = 64
NUM_WORKERS = 4

# Debugging
# Set to True to process a small subset of data for quick pipeline verification
DEBUG = False
DEBUG_SAMPLES = 100

# ==========================================
# Model Configuration
# ==========================================
# Backbone
MODEL_NAME = "convnext_large"
# Using "New Recipe" weights as specified in strategy
WEIGHTS = "IMAGENET1K_V1"

# ==========================================
# Multi-Scale Preprocessing Pipelines
# ==========================================
# Defines the three views for the ensemble.
# 'resize': int (resize smaller edge) or tuple (h, w) (resize strictly)
# 'crop_size': int (size of center crop) or None (no cropping)
SCALE_CONFIGS = {
    "standard": {
        "resize": 232,
        "crop_size": 224,
        "description": "Balance: Resize smaller edge to 232, Center Crop 224",
    },
    "global": {
        "resize": (224, 224),
        "crop_size": None,
        "description": "Shape: Resize strictly to 224x224, No Crop (Squish)",
    },
    "local": {
        "resize": 288,
        "crop_size": 224,
        "description": "Texture: Resize smaller edge to 288, Center Crop 224 (Zoom)",
    },
}

# ==========================================
# Classifier Configuration
# ==========================================
# Logistic Regression Hyperparameters
LOGREG_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "C": 1.0,  # Regularization strength (inverse)
    "n_jobs": -1,
    "random_state": SEED,
}
