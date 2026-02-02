import os
import torch


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    # Root directory for input data (Read-Only)
    INPUT_DIR = "./input"

    # Directory containing the generated metadata CSVs
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching intermediate files (features, models)
    # Using 'idea_27' as specified for this experiment
    WORKING_DIR = "./working/idea_27"

    # Output path for the final submission file
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Global Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # 3. Data Processing & Anchoring
    # ==========================================
    # Image size for backbone models (Standard ImageNet resolution)
    IMAGE_SIZE = 224

    # Normalization constants (ImageNet defaults)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # Moment-Aligned Canonical Anchoring settings
    # We use 4 orthogonal views after moment-alignment
    ANCHOR_ANGLES = [0, 90, 180, 270]

    # ==========================================
    # 4. Model Architectures (Dual-Stream)
    # ==========================================
    # Global Geometry Stream: DINOv2 ViT-Large
    # Using timm registry name
    MODEL_DINO = "vit_large_patch14_dinov2.lvd142m"

    # Local Texture Stream: ConvNeXt Large
    # Using timm registry name
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"

    # ==========================================
    # 5. Feature Engineering
    # ==========================================
    # PCA Variance retention threshold for visual features
    PCA_VARIANCE = 0.99

    # Tabular features to include (prefixes based on dataset)
    TABULAR_PREFIXES = ["margin", "shape", "texture"]

    # ==========================================
    # 6. Training & Validation
    # ==========================================
    # Cross-Validation Strategy
    N_FOLDS = 10

    # Classifier Settings (LDA with Ledoit-Wolf)
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"

    # Hardware settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32  # Safe for A100 with Large models
    NUM_WORKERS = 4
