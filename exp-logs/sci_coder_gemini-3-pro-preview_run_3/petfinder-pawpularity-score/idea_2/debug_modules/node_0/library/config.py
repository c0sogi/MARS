import os
import torch

# ==========================================
# 1. Global Configuration
# ==========================================
SEED = 42
DEBUG = False  # Set to True to use a subset of data for debugging

# ==========================================
# 2. File Paths & Directories
# ==========================================
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_meta.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_meta.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_meta.csv")

# Submission File
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Files (for feature extraction)
# We use .npy files for efficient storage of numpy arrays
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.npy")
TRAIN_TARGETS_PATH = os.path.join(WORKING_DIR, "train_targets.npy")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.npy")
VAL_TARGETS_PATH = os.path.join(WORKING_DIR, "val_targets.npy")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.npy")
TEST_IDS_PATH = os.path.join(WORKING_DIR, "test_ids.npy")

# Model Save Path
SVR_MODEL_PATH = os.path.join(WORKING_DIR, "svr_model.joblib")

# ==========================================
# 3. Data & Image Processing
# ==========================================
IMG_SIZE = 224
BATCH_SIZE = 64  # Batch size for feature extraction
NUM_WORKERS = 4  # Number of DataLoader workers

# ImageNet Normalization Statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Metadata Features (Binary columns to include)
META_FEATURES = [
    "Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]

# ==========================================
# 4. Model Architectures
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Backbone Models (timm library names)
# Swin Transformer for global composition
BACKBONE_SWIN = "swin_base_patch4_window7_224"
# EfficientNetV2 for high-frequency local details
BACKBONE_EFFNET = "tf_efficientnetv2_s"

# ==========================================
# 5. SVR Hyperparameters
# ==========================================
# Grid for GridSearchCV
SVR_GRID = {
    "C": [0.1, 1.0, 10.0, 50.0],
    "epsilon": [0.01, 0.1, 0.5, 1.0, 2.0],
    "kernel": ["rbf"],
    "gamma": ["scale", "auto"],
}

# Cross-Validation Folds
N_FOLDS = 5
