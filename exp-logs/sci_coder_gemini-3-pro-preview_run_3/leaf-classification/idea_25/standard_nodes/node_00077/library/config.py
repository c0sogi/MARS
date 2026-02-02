import os
import torch
import numpy as np


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    # Root directory for input data (Read-Only)
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Directory for pre-generated metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching intermediate files (Write-Allowed)
    # Using specific subdirectory for this idea to avoid conflicts
    WORKING_DIR = "./working/idea_25"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Output path for final submission
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # 2. Global Settings
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # ==========================================
    # 3. Data Processing & Manifold Densification
    # ==========================================
    IMAGE_SIZE = 224

    # 12 Equidistant views: 0, 30, 60, ..., 330
    NUM_VIEWS = 12
    ROTATION_ANGLES = (
        np.linspace(0, 360, NUM_VIEWS, endpoint=False).astype(int).tolist()
    )

    # Orthogonal Centroids Configuration
    # We group the 12 views into 3 centroids of 4 orthogonal views each.
    # Indices correspond to the ROTATION_ANGLES list.
    # Centroid A: 0, 90, 180, 270 (Indices: 0, 3, 6, 9)
    # Centroid B: 30, 120, 210, 300 (Indices: 1, 4, 7, 10)
    # Centroid C: 60, 150, 240, 330 (Indices: 2, 5, 8, 11)
    NUM_CENTROIDS = 3
    CENTROID_INDICES = [
        [0, 3, 6, 9],  # Centroid A
        [1, 4, 7, 10],  # Centroid B
        [2, 5, 8, 11],  # Centroid C
    ]

    # ==========================================
    # 4. Feature Extraction Models
    # ==========================================
    BATCH_SIZE_EXTRACTION = 32

    # Global Geometry Stream: DINOv2 (ViT-Large)
    # Using HuggingFace Transformers ID
    MODEL_DINO_ID = "facebook/dinov2-large"

    # Local Texture Stream: ConvNeXt Large
    # Using HuggingFace Transformers ID
    MODEL_CONVNEXT_ID = "facebook/convnext-large-224-22k-1k"

    # ==========================================
    # 5. Pipeline Hyperparameters
    # ==========================================
    # Cross-Validation
    N_FOLDS = 5

    # Dimensionality Reduction
    # Retain 99% variance for visual features (Linear Topology)
    PCA_VARIANCE = 0.99

    # Tabular Transformation
    # Quantile Transformer output distribution
    TABULAR_TRANSFORM_DIST = "normal"

    # Classifier
    # LDA Solver
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf
