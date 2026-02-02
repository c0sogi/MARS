import os
import torch

# ==========================================
# Global Configuration
# ==========================================

# Random Seed for reproducibility across all libraries
SEED = 42

# Compute Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Optimized for the 12 vCPUs available

# ==========================================
# Directory Paths
# ==========================================

# Input Data (Read-Only)
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Metadata CSV Files
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Working Directory for Caching (Idea 9)
# Stores intermediate embeddings to enable rapid experimentation without re-inference
WORKING_DIR = "./working/idea_9"
os.makedirs(WORKING_DIR, exist_ok=True)

# Output Directory for Final Submission
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Model Architecture Configuration
# ==========================================

# Backbone: ConvNeXt-Large
MODEL_NAME = "convnext_large"
MODEL_WEIGHTS = "IMAGENET1K_V1"  # Torchvision 'New Recipe' weights

# Feature Pyramid Extraction
# We extract features from the final stage (semantic) and the penultimate stage (texture)
# Mapping torchvision layer names to logical names
FEATURE_LAYERS = {
    "features.5": "stage3",  # Intermediate layer (Texture/Pattern)
    "features.7": "stage4",  # Final layer (Global Shape/Semantic)
}

# Embedding Dimensions (ConvNeXt-Large specific)
# Used for calculating final concatenated vector size
EMBEDDING_DIMS = {"stage3": 768, "stage4": 1536}

# ==========================================
# Data Preprocessing / Views Configuration
# ==========================================

# View 1: Global (Squish)
# Resizes image to square, distorting aspect ratio but preserving full content
GLOBAL_VIEW_SIZE = 224

# View 2: Standard (Crop)
# Traditional resize-then-crop approach matching pre-training
STANDARD_VIEW_RESIZE = 232
STANDARD_VIEW_CROP = 224

# View 3: Robust Local (FiveCrop)
# High-res resize followed by 5 crops (corners + center) to capture details
LOCAL_VIEW_RESIZE = 288
LOCAL_VIEW_CROP = 224
LOCAL_VIEW_NUM_CROPS = 5

# ImageNet Normalization Constants
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ==========================================
# Caching Configuration
# ==========================================

# File paths for caching processed features, labels, and IDs
# Organized by dataset split
CACHE_FILES = {
    "train": {
        "features": os.path.join(WORKING_DIR, "train_features.npy"),
        "labels": os.path.join(WORKING_DIR, "train_labels.npy"),
        "ids": os.path.join(WORKING_DIR, "train_ids.npy"),
    },
    "val": {
        "features": os.path.join(WORKING_DIR, "val_features.npy"),
        "labels": os.path.join(WORKING_DIR, "val_labels.npy"),
        "ids": os.path.join(WORKING_DIR, "val_ids.npy"),
    },
    "test": {
        "features": os.path.join(WORKING_DIR, "test_features.npy"),
        "ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    },
}

# ==========================================
# Training Hyperparameters
# ==========================================

BATCH_SIZE = 32

# LogisticRegressionCV Configuration
# Using L-BFGS solver for efficient multiclass optimization
LOGREG_PARAMS = {
    "Cs": 10,  # Grid of 10 values for regularization parameter C
    "cv": 5,  # 5-fold internal cross-validation
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 2000,  # High iteration count to ensure convergence
    "class_weight": "balanced",  # Handle class imbalance
    "n_jobs": -1,  # Use all available cores
    "random_state": SEED,
    "verbose": 0,
}
