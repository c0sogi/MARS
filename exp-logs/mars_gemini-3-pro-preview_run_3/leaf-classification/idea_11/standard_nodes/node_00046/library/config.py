import os
import torch


class Config:
    """
    Configuration class for the Leaf Classification Task.
    Implements settings for 'View-Expanded Manifold Stabilization for Linear Discriminant Analysis'.
    """

    # ==========================================
    # Global Constants & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLES = 50  # Number of samples to use when DEBUG is True

    # ==========================================
    # Directory & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working directory for caching intermediate artifacts (Idea 11)
    WORKING_DIR = "./working/idea_11"

    # Output directory for final submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architectures & Image Settings
    # ==========================================
    # Global Geometry Stream: DINOv2 (ViT-Large)
    # Using timm model name
    DINO_MODEL_NAME = "vit_large_patch14_dinov2.lvd142m"
    DINO_IMG_SIZE = 518

    # Local Texture Stream: ConvNeXt Large
    # Using timm model name
    CONVNEXT_MODEL_NAME = "convnext_large.fb_in22k_ft_in1k"
    CONVNEXT_IMG_SIZE = 384

    # ==========================================
    # Compute & Processing
    # ==========================================
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Pipeline Hyperparameters
    # ==========================================
    # Dimensionality Reduction
    PCA_VARIANCE = 0.99

    # Classification (LDA with Ledoit-Wolf Shrinkage)
    LDA_SOLVER = "lsqr"  # 'lsqr' or 'eigen' required for shrinkage
    LDA_SHRINKAGE = "auto"  # Automatic Ledoit-Wolf shrinkage

    # Validation
    N_FOLDS = 5

    @classmethod
    def setup(cls):
        """
        Initialize the directory structure for outputs and cache.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
