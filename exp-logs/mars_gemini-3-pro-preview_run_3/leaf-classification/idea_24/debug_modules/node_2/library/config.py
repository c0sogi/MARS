import os
import torch

# ==========================================
# 1. Directory & File Paths
# ==========================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_24"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Input subdirectories
IMAGES_DIR = os.path.join(INPUT_DIR, "images")

# Metadata files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cached Feature Paths (for deterministic caching)
# We store raw extracted features (before expert splitting) here
CACHE_TRAIN_IMG_FEATURES = os.path.join(WORKING_DIR, "train_img_features_36views.npy")
CACHE_TEST_IMG_FEATURES = os.path.join(WORKING_DIR, "test_img_features_36views.npy")
CACHE_TRAIN_TAB_FEATURES = os.path.join(WORKING_DIR, "train_tab_features.npy")
CACHE_TEST_TAB_FEATURES = os.path.join(WORKING_DIR, "test_tab_features.npy")
CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "train_ids.npy")
CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")
CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")
CACHE_CLASSES = os.path.join(WORKING_DIR, "classes.npy")

# Submission Output
SUBMISSION_PATH = "./submission/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# ==========================================
# 2. Global Hyperparameters
# ==========================================
SEED = 42
N_FOLDS = 10  # Stratified K-Fold
BATCH_SIZE = 32
NUM_WORKERS = 4

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 3. Data Processing & Augmentation
# ==========================================
# Image Parameters
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Rotational Manifold Settings
N_ROTATIONS = 36  # Extract features for 36 equidistant views (0, 10, ..., 350)
N_EXPERTS = 9  # Number of orthogonal experts in the ensemble
VIEWS_PER_CENTROID = N_ROTATIONS // N_EXPERTS  # 4 views per centroid

# Tabular Feature Prefixes
TABULAR_PREFIXES = ["margin", "shape", "texture"]
N_TABULAR_FEATURES = 192  # 64 * 3

# ==========================================
# 4. Model Architecture & Feature Extraction
# ==========================================
# Feature Extractors (timm model names)
# DINOv2 for global geometry
MODEL_DINO = "vit_large_patch14_dinov2"
# ConvNeXt for local texture/margin details
MODEL_CONVNEXT = "convnext_large"

# Dimensionality Reduction & Topology
PCA_VARIANCE = 0.99  # Retain 99% variance for visual features
QT_OUTPUT_DIST = "normal"  # Quantile Transformer output distribution for tabular data

# ==========================================
# 5. Training Configuration
# ==========================================
# LDA Solver settings
LDA_SOLVER = "lsqr"
LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage


# Utility to print config
def print_config():
    print("=" * 40)
    print("       CONFIGURATION SUMMARY")
    print("=" * 40)
    print(f"Working Directory: {WORKING_DIR}")
    print(f"Device:            {DEVICE}")
    print(f"Seed:              {SEED}")
    print(f"Folds:             {N_FOLDS}")
    print("-" * 40)
    print(f"Rotations:         {N_ROTATIONS}")
    print(f"Experts:           {N_EXPERTS}")
    print(f"Views/Expert:      {VIEWS_PER_CENTROID}")
    print("-" * 40)
    print(f"DINO Model:        {MODEL_DINO}")
    print(f"ConvNeXt Model:    {MODEL_CONVNEXT}")
    print(f"PCA Variance:      {PCA_VARIANCE}")
    print("=" * 40)
