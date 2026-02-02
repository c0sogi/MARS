import os


class Config:
    """
    Central configuration module for the Leaf Species Classification task.
    Implements settings for 'Selective-Topology Orthogonal Manifold-Densified LDA'.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42

    # ==========================================
    # Data Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata Paths (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # Output & Caching
    # ==========================================
    # Working directory for caching intermediate features (parquet/npy)
    WORKING_DIR = "./working/idea_30"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architectures
    # ==========================================
    # Global Geometry Stream: DINOv2 (ViT-Large)
    # Captures self-supervised geometric priors
    MODEL_DINO = "vit_large_patch14_dinov2"

    # Local Texture Stream: ConvNeXt Large
    # Captures high-frequency margin details
    MODEL_CONVNEXT = "convnext_large"

    # ==========================================
    # Preprocessing & Augmentation
    # ==========================================
    IMG_SIZE = 224

    # Multi-View Extraction: 12 equidistant rotations
    # Angles: 0, 30, 60, ..., 330
    NUM_ROTATIONS = 12

    # Manifold Densification
    # We generate 3 distinct "Orthogonal Centroids" per image
    # Each centroid is the average of 4 orthogonal views
    NUM_CENTROIDS = 3
    VIEWS_PER_CENTROID = 4

    # ==========================================
    # Feature Engineering
    # ==========================================
    # PCA Variance Retention for Visual Streams
    # Preserves 99% of variance to maintain linear separability
    PCA_VARIANCE = 0.99

    # Tabular Feature Groups
    TABULAR_PREFIXES = ["margin", "shape", "texture"]
    NUM_TABULAR_FEATURES = 192  # 64 * 3

    # ==========================================
    # Training & Validation
    # ==========================================
    # Stratified K-Fold Cross-Validation
    N_FOLDS = 5

    # DataLoader Configuration
    BATCH_SIZE = 4
    NUM_WORKERS = 4

    # ==========================================
    # Utilities
    # ==========================================
    @staticmethod
    def make_dirs():
        """Creates necessary working and submission directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
