import os
import torch
import numpy as np


class Config:
    """
    Global configuration for the Leaf Species Identification task.
    Implements settings for Stacked Discriminant Analysis with Hyper-Densified OOF Projection.
    """

    # ==========================================
    # 1. General Settings
    # ==========================================
    SEED = 42
    NUM_FOLDS = 10  # Stratified K-Fold for Outer Evaluation
    INNER_FOLDS = 5  # K-Fold for Inner Meta-Feature Generation (OOF Stacking)

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    # Input Data
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
    TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    METADATA_TRAIN = os.path.join(METADATA_DIR, "train.csv")
    METADATA_VAL = os.path.join(METADATA_DIR, "val.csv")
    METADATA_TEST = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory (Outputs)
    WORKING_DIR = "./working/idea_17"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODELS_DIR = os.path.join(WORKING_DIR, "models")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Data Processing & Hyper-Densification
    # ==========================================
    # Image Parameters
    IMAGE_SIZE = 224  # Standard input size for ViT/ConvNeXt

    # Rotation Augmentation
    NUM_ROTATIONS = 36
    ROTATION_STEP = 360 // NUM_ROTATIONS  # 10 degrees
    ROTATION_ANGLES = list(range(0, 360, ROTATION_STEP))

    # Hyper-Densification Topology (Training)
    # 9 Centroids * 4 Views = 36 Total Views
    NUM_TRAIN_CENTROIDS = 9
    VIEWS_PER_CENTROID = 4
    # Logic: Centroid k averages indices [k, k+9, k+18, k+27] corresponding to orthogonal angles

    # Inference Topology (Validation/Test)
    # Single Canonical Centroid: Average of 0, 90, 180, 270 degrees
    CANONICAL_ANGLES = [0, 90, 180, 270]

    # Tabular Features
    TABULAR_PREFIXES = ["margin", "shape", "texture"]
    NUM_TABULAR_FEATURES = 192  # 64 * 3

    # ==========================================
    # 4. Model Hyperparameters
    # ==========================================
    # Feature Extractors (timm model names)
    # DINOv2 ViT-Large
    MODEL_DINO = "vit_large_patch14_dinov2.lvd142m"
    # ConvNeXt Large
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"

    BATCH_SIZE_EXTRACTION = 32

    # Dimensionality Reduction
    PCA_VARIANCE = 0.99  # Retain 99% variance

    # Linear Discriminant Analysis
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

    # Meta-Learner
    META_SOLVER = "lsqr"
    META_SHRINKAGE = "auto"

    @classmethod
    def setup_directories(cls):
        """Creates necessary working directories."""
        dirs = [cls.WORKING_DIR, cls.CACHE_DIR, cls.MODELS_DIR, cls.SUBMISSION_DIR]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def get_centroid_indices(cls, centroid_idx):
        """
        Returns the indices of the rotation angles that belong to a specific
        training centroid k (0..8).

        Logic: To get orthogonal views (90 deg apart), we take stride = NUM_ROTATIONS // 4.
        """
        stride = cls.NUM_ROTATIONS // cls.VIEWS_PER_CENTROID
        indices = [centroid_idx + (i * stride) for i in range(cls.VIEWS_PER_CENTROID)]
        return indices

    @classmethod
    def get_canonical_indices(cls):
        """
        Returns indices corresponding to the canonical angles [0, 90, 180, 270].
        """
        indices = []
        for angle in cls.CANONICAL_ANGLES:
            if angle in cls.ROTATION_ANGLES:
                indices.append(cls.ROTATION_ANGLES.index(angle))
            else:
                # Fallback if exact angle not in grid (unlikely with step 10)
                closest_idx = np.argmin(np.abs(np.array(cls.ROTATION_ANGLES) - angle))
                indices.append(closest_idx)
        return indices
