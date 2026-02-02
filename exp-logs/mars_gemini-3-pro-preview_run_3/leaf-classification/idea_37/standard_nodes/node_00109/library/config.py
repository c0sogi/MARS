import os
import torch


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this experiment iteration
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_37")
    SUBMISSION_DIR = "./submission"

    # File paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # ==========================================
    # Global Seeding
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Image preprocessing
    IMAGE_SIZE = 224  # Standard input size for ConvNeXt and DINOv2
    BATCH_SIZE = 32
    NUM_WORKERS = 2  # Conservative number of workers for data loading

    # Multi-View Extraction Strategy
    N_VIEWS = 12  # Total rotations per image (0, 30, 60, ..., 330)

    # Manifold Densification (Orthogonal Centroids)
    # We split 12 views into 3 centroids of 4 orthogonal views each
    N_CENTROIDS = 3
    VIEWS_PER_CENTROID = 4

    # ==========================================
    # Model Architecture
    # ==========================================
    # Feature Extractors (timm model names)
    # DINOv2 ViT-Large for global geometry
    MODEL_DINOV2 = "vit_large_patch14_dinov2.lvd142m"

    # ConvNeXt Large for local texture/margins
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Dimensionality Reduction & Classifier
    # ==========================================
    # PCA Variance retention for visual streams
    PCA_VARIANCE = 0.99

    # Quantile Transformer output distribution for tabular data
    TABULAR_OUTPUT_DIST = "normal"

    # Classifier Settings
    CLASSIFIER_TYPE = "lda"
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

    # ==========================================
    # Training & Validation
    # ==========================================
    N_FOLDS = 10
    STRATIFIED = True

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for caching and submission.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Print configuration summary
        print(f"Configuration Setup Complete.")
        print(f"  Cache Directory: {cls.CACHE_DIR}")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Model DINOv2: {cls.MODEL_DINOV2}")
        print(f"  Model ConvNeXt: {cls.MODEL_CONVNEXT}")
        print(f"  Folds: {cls.N_FOLDS}")
