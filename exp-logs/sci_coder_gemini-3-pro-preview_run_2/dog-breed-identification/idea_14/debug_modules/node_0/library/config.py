import os
import numpy as np

# ==========================================
# Global Configuration & Constants
# ==========================================

# Random Seed for Reproducibility
SEED = 42

# ------------------------------------------
# Directory Paths
# ------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for this idea to ensure cache safety and isolation
WORKING_DIR = "./working/idea_14"
SUBMISSION_DIR = "./submission"

# Create directories if they don't exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ------------------------------------------
# File Paths
# ------------------------------------------
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
LABELS_PATH = os.path.join(INPUT_DIR, "labels.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ------------------------------------------
# Model Architecture Configurations
# ------------------------------------------
# Stream A: Supervised CNN (ConvNeXt-Large)
# Using timm identifier corresponding to the requested architecture.
STREAM_A_MODEL_NAME = "convnext_large.fb_in1k"
STREAM_A_CACHE_PREFIX = "convnext_large_sup"

# Stream B: Self-Supervised ViT (DINOv2 ViT-Large)
# Using timm identifier for DINOv2.
STREAM_B_MODEL_NAME = "vit_large_patch14_dinov2.lvd142m"
STREAM_B_CACHE_PREFIX = "vit_large_dinov2"

# ------------------------------------------
# Data Processing & View Generation
# ------------------------------------------
# Target Input Size for Models
IMG_SIZE = 224

# View 1: Global
# Strategy: Squish to (224, 224) - handled in transforms

# View 2: Standard
# Strategy: Resize to 232, Center Crop to 224
RESIZE_STANDARD = 232
CROP_STANDARD = 224

# View 3: Local
# Strategy: Resize to 288, Center Crop to 224 (Zoom)
RESIZE_LOCAL = 288
CROP_LOCAL = 224

# Normalization Constants (Standard ImageNet)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ------------------------------------------
# Compute & Training Hyperparameters
# ------------------------------------------
# Hardware settings
NUM_WORKERS = 4
DEVICE = "cuda"

# Batch Size (Optimized for A100 40GB)
BATCH_SIZE = 64

# Logistic Regression Head Hyperparameters
# Using a logarithmic scale for regularization parameter C
LOGREG_C_VALUES = np.logspace(-4, 4, 20)
LOGREG_CV_FOLDS = 5
LOGREG_MAX_ITER = 1000
LOGREG_JOBS = -1

# ------------------------------------------
# Execution Flags
# ------------------------------------------
# Controls whether to load pre-computed embeddings from WORKING_DIR
LOAD_CACHED_DATA = True
