import os
import numpy as np

# ==========================================
# Project Directory Structure
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific cache directory for this experiment (Idea 39)
# Used to store extracted features, PCA models, and intermediate arrays
CACHE_DIR = os.path.join(WORKING_DIR, "idea_39")

# Ensure necessary writeable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Reproducibility
# ==========================================
SEED = 42

# ==========================================
# Data Processing & Augmentation
# ==========================================
# Image Input Specification
IMG_SIZE = 224  # Standard input size for ViT and ConvNeXt
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Rotation Settings for Multi-View Extraction
# We extract 12 equidistant views to capture the leaf shape manifold
NUM_ROTATIONS = 12
ROTATION_ANGLES = np.linspace(0, 360, NUM_ROTATIONS, endpoint=False).tolist()
# Resulting angles: [0.0, 30.0, 60.0, 90.0, ..., 330.0]

# Manifold Densification (Orthogonal Centroids)
# We aggregate views into 3 orthogonal centroids to increase effective N
# while reducing variance.
# Indices correspond to the ROTATION_ANGLES list.
CENTROID_INDICES = {
    # Centroid A: 0°, 90°, 180°, 270°
    "A": [0, 3, 6, 9],
    # Centroid B: 30°, 120°, 210°, 300°
    "B": [1, 4, 7, 10],
    # Centroid C: 60°, 150°, 240°, 330°
    "C": [2, 5, 8, 11],
}

# ==========================================
# Model Architecture
# ==========================================
# Using timm library naming conventions
# Global Geometry Stream: DINOv2 ViT-Large
MODEL_DINO_NAME = "vit_large_patch14_dinov2.lvd142m"

# Local Texture Stream: ConvNeXt Large
MODEL_CONVNEXT_NAME = "convnext_large.fb_in22k_ft_in1k"

# Inference Parameters
BATCH_SIZE = 32
NUM_WORKERS = 4  # Tuned for 12 vCPUs

# ==========================================
# Feature Engineering & Dimensionality Reduction
# ==========================================
# Independent Subspace Reduction
# We retain 99% variance for each visual stream independently
PCA_VARIANCE_THRESHOLD = 0.99

# Tabular Data Configuration
# The 3 sets of handcrafted features provided in the dataset
TABULAR_PREFIXES = ["margin", "shape", "texture"]
TOTAL_TABULAR_FEATURES = 192  # 64 * 3

# ==========================================
# Classification & Training
# ==========================================
# Cross-Validation Strategy
N_FOLDS = 5
STRATIFIED = True

# Linear Discriminant Analysis (LDA) Hyperparameters
# Using Ledoit-Wolf shrinkage ('auto') with Least Squares solver
LDA_SOLVER = "lsqr"
LDA_SHRINKAGE = "auto"

# ==========================================
# Debugging & Development
# ==========================================
# Set to an integer (e.g., 100) to limit the dataset size for rapid prototyping.
# Set to None for full training.
DEBUG_SAMPLE_SIZE = None
