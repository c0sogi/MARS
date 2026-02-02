import os


class Config:
    """
    Configuration for Stratified Selective-Topology Orthogonal Manifold-Densified LDA
    with Global Variance Alignment.
    """

    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for caching intermediate artifacts (idea_33)
    WORKING_DIR = "./working/idea_33"
    SUBMISSION_DIR = "./submission"

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Image Source Directory
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Output Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Reproducibility & Debugging
    # ==========================================
    SEED = 42

    # Debugging flags to control dataset size for rapid iteration
    DEBUG = False
    DEBUG_LIMIT = 50  # Number of samples to use when DEBUG is True

    # ==========================================
    # Model Architecture & Feature Extraction
    # ==========================================
    # Global Geometry Stream (ViT)
    MODEL_DINO = "facebook/dinov2-large"

    # Local Texture Stream (ConvNeXt)
    MODEL_CONVNEXT = "facebook/convnext-large-224-22k"

    # Image Input Specifications
    IMG_SIZE = 224
    # ImageNet Normalization Constants
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # ==========================================
    # Manifold Densification (Orthogonal Views)
    # ==========================================
    # 12 Equidistant Rotations (0 to 330 degrees)
    ROTATION_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

    # Orthogonal Centroids Definition
    # Maps centroid names to indices in ROTATION_ANGLES
    # Centroid A: {0, 90, 180, 270}   -> Indices [0, 3, 6, 9]
    # Centroid B: {30, 120, 210, 300} -> Indices [1, 4, 7, 10]
    # Centroid C: {60, 150, 240, 330} -> Indices [2, 5, 8, 11]
    CENTROID_INDICES = {"A": [0, 3, 6, 9], "B": [1, 4, 7, 10], "C": [2, 5, 8, 11]}

    # ==========================================
    # Feature Engineering & Topology
    # ==========================================
    # Independent Subspace Reduction (Visual Streams)
    PCA_VARIANCE = 0.99

    # Tabular Gaussianization
    # 192 features: 64 margin + 64 shape + 64 texture
    TABULAR_COLS_PREFIXES = ["margin", "shape", "texture"]
    TABULAR_TRANSFORM_METHOD = (
        "quantile_normal"  # QuantileTransformer(output_distribution='normal')
    )

    # Global Variance Alignment
    GLOBAL_SCALER = "standard"  # StandardScaler applied to concatenated features

    # ==========================================
    # Training Strategy
    # ==========================================
    # Stratified K-Fold Ensemble
    N_FOLDS = 10

    # Classifier Configuration (LDA with Ledoit-Wolf Shrinkage)
    CLASSIFIER_TYPE = "LDA"
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # 'auto' triggers Ledoit-Wolf lemma

    # ==========================================
    # Post-Processing
    # ==========================================
    # Probability clipping to avoid log-loss extremes
    PROB_CLIP_MIN = 1e-15
    PROB_CLIP_MAX = 1.0 - 1e-15
