import os
import torch
from torchvision.transforms import InterpolationMode

# ==========================================
# Global Paths & Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_17"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
LABELS_PATH = os.path.join(INPUT_DIR, "labels.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Reproducibility & Compute
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12  # Optimized for 12 vCPUs
BATCH_SIZE = 64  # Efficient for A100 40GB (Inference/Feature Extraction)

# ==========================================
# Model Architecture
# ==========================================
MODEL_NAME = "convnext_large"
WEIGHTS = "IMAGENET1K_V1"
EMBEDDING_DIM = 1536  # Native feature dimension for ConvNeXt-Large

# ==========================================
# Data Processing & Multi-View Strategy
# ==========================================
# Common Settings
IMAGE_SIZE = 224
INTERPOLATION = InterpolationMode.BICUBIC

# View 1: Global (Shape) - Squish to 224x224
VIEW_GLOBAL_SIZE = (224, 224)

# View 2: Standard (Context) - Resize 232 -> CenterCrop 224
VIEW_STANDARD_RESIZE = 232
VIEW_STANDARD_CROP = 224

# View 3: Local (Texture) - Resize 288 -> CenterCrop 224
VIEW_LOCAL_RESIZE = 288
VIEW_LOCAL_CROP = 224

# Test Time Augmentation
USE_TTA_FLIP = True  # Horizontal Flip

# ==========================================
# Training & Head Hyperparameters
# ==========================================
# Feature Fusion
# 3 Views * 1536 dim = 4608 features
TOTAL_FEATURE_DIM = EMBEDDING_DIM * 3

# Linear Head: Logistic Regression CV
LOGREG_PARAMS = {
    "cv": 5,
    "max_iter": 2000,
    "solver": "lbfgs",
    "n_jobs": -1,
    "random_state": SEED,
    "class_weight": "balanced",
}

# Non-Linear Head: Histogram Gradient Boosting
GB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 500,
    "max_depth": 8,
    "validation_fraction": 0.15,
    "n_iter_no_change": 20,
    "early_stopping": True,
    "random_state": SEED,
    "verbose": 0,
}
