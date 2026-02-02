import os
import numpy as np


class Config:
    """
    Central configuration for the Leaf Classification project.
    Implements settings for 'Convex-Hull Densified Selective-Topology LDA'.
    """

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_40"
    SUBMISSION_DIR = "./submission"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache File Paths (Parquet/Numpy for intermediate storage)
    # We use these to store the raw 12-view features before densification
    TRAIN_FEATURES_CACHE = os.path.join(WORKING_DIR, "train_features_12view.parquet")
    VAL_FEATURES_CACHE = os.path.join(WORKING_DIR, "val_features_12view.parquet")
    TEST_FEATURES_CACHE = os.path.join(WORKING_DIR, "test_features_12view.parquet")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Global Hyperparameters
    # ==========================================
    SEED = 42
    N_FOLDS = 5  # Stratified K-Fold
    NUM_WORKERS = 4

    # ==========================================
    # Visual Feature Extraction Settings
    # ==========================================
    # Input Image Size
    IMAGE_SIZE = 224

    # Batch Size for Inference/Extraction
    BATCH_SIZE = 4

    # Rotation Strategy: 12 Equidistant Views (0, 30, ..., 330)
    # Used to generate the convex hull of the manifold
    ROTATION_ANGLES = list(range(0, 360, 30))

    # Model Architectures (timm library names)
    # Global Geometry Stream: DINOv2 Large
    MODEL_DINO_NAME = "vit_large_patch14_dinov2.lvd142m"

    # Local Texture Stream: ConvNeXt Large
    MODEL_CONVNEXT_NAME = "convnext_large.fb_in22k_ft_in1k"

    # ==========================================
    # Pipeline / Model Parameters
    # ==========================================
    # Independent Subspace Reduction (PCA)
    PCA_VARIANCE = 0.99

    # Tabular Data
    TABULAR_PREFIXES = ["margin", "shape", "texture"]
    N_TABULAR_FEATURES = 192  # 64 * 3

    # LDA Classifier Settings
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

    # ==========================================
    # Utility Methods
    # ==========================================
    @classmethod
    def setup(cls):
        """
        Ensures necessary working directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
