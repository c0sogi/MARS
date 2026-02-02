import os


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate files (features, models)
    # Using 'idea_29' as specified for this iteration
    WORKING_DIR = "./working/idea_29"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Global Hyperparameters
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4  # For data loading

    # ==========================================
    # Feature Extraction Hyperparameters
    # ==========================================
    # Multi-View Extraction: 12 equidistant rotations
    # Angles: 0, 30, 60, ..., 330
    N_ROTATIONS = 12
    BATCH_SIZE = 4

    # Models
    # Using 'timm' names or standard identifiers
    MODEL_DINO = "vit_large_patch14_dinov2.lvd142m"  # DINOv2 ViT-Large
    MODEL_CONV = "convnext_large.fb_in22k_ft_in1k"  # ConvNeXt Large

    # ==========================================
    # Data Processing & Manifold Densification
    # ==========================================
    # Orthogonal Centroids: 3 centroids per image, each averaging 4 views
    # Centroid A: {0, 90, 180, 270}
    # Centroid B: {30, 120, 210, 300}
    # Centroid C: {60, 150, 240, 330}
    N_CENTROIDS = 3
    VIEWS_PER_CENTROID = 4

    # Dimensionality Reduction (Independent Subspace Reduction)
    # Retain 99% variance for visual streams
    PCA_VARIANCE = 0.99

    # Tabular Processing
    # Quantile Transformer output distribution
    TABULAR_OUTPUT_DIST = "normal"

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Cross-Validation
    N_FOLDS = 10

    # Classifier Settings (LDA)
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Create subdirectories for specific caches to keep things organized
        os.makedirs(os.path.join(cls.WORKING_DIR, "features"), exist_ok=True)
        os.makedirs(os.path.join(cls.WORKING_DIR, "models"), exist_ok=True)


# Execute setup immediately when module is imported to ensure paths exist
Config.setup()
