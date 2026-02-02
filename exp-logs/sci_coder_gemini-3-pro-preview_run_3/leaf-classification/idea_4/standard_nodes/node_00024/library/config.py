import os

# ==========================================
# Global Path Configuration
# ==========================================
INPUT_DIR = "./input"
IMAGES_DIR = os.path.join(INPUT_DIR, "images")
METADATA_DIR = "./metadata"

# Metadata Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Directories
WORKING_DIR = "./working/idea_4"
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Model & Feature Configuration
# ==========================================
# Model Architecture Names (timm compatible)
# Model 1: Self-Supervised Geometric Features (DINOv2 ViT-Large)
MODEL_1_NAME = "vit_large_patch14_dinov2"

# Model 2: High-Frequency Texture Features (ConvNeXt Large)
MODEL_2_NAME = "convnext_large"

# Image Resolution
# Using 224x224 as a standard resolution compatible with both backbones.
# While DINOv2 often uses 518, 224 is sufficient for binary leaf shapes
# and aligns with standard ConvNeXt pretraining.
IMG_SIZE = 224

# Dimensionality Reduction
# Variance threshold for PCA to retain discriminative signals in the tail
PCA_VARIANCE = 0.99

# ==========================================
# Training & Optimization Configuration
# ==========================================
# Random Seed for Reproducibility
SEED = 42

# DataLoader settings
BATCH_SIZE = 128
NUM_WORKERS = 8

# Ensemble Optimization
# Step size for the discrete grid search of the mixing weight 'w'
GRID_SEARCH_STEP = 0.01

# Feature Groups (Tabular)
# Defined here for easy access by feature processing modules
MARGIN_COLS_PREFIX = "margin"
SHAPE_COLS_PREFIX = "shape"
TEXTURE_COLS_PREFIX = "texture"
