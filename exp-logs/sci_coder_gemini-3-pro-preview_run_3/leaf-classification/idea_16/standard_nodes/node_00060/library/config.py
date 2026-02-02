import os


class Config:
    """
    Configuration for Hierarchical Discriminant Stacking with Hyper-Densified Orthogonal Centroids.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    N_CLASSES = 99

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "models")
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing / Densification
    # ==========================================
    # Rotation logic: 36 views from 0 to 350 degrees
    ROTATION_STEP = 10
    ROTATION_ANGLES = list(range(0, 360, ROTATION_STEP))
    NUM_VIEWS = len(ROTATION_ANGLES)  # 36

    # Hyper-Densification: 9 centroids per image, each composed of 4 orthogonal views
    # Views per centroid
    VIEWS_PER_CENTROID = 4
    # Number of training centroids per image
    N_CENTROIDS_TRAIN = NUM_VIEWS // VIEWS_PER_CENTROID  # 9

    # Canonical Centroid Indices for Inference (0, 90, 180, 270 degrees)
    # Indices correspond to the position in ROTATION_ANGLES
    CANONICAL_INDICES = [0, 9, 18, 27]

    # ==========================================
    # Model Architectures & Hyperparameters
    # ==========================================
    # Feature Extractors (timm model names)
    # DINOv2 Large (ViT-L/14)
    DINO_MODEL_NAME = "vit_large_patch14_dinov2.lvd142m"
    # ConvNeXt Large
    CONVNEXT_MODEL_NAME = "convnext_large.fb_in22k_ft_in1k"

    # Image Input Sizes
    IMG_SIZE = 224

    # Dimensionality Reduction
    PCA_VARIANCE = 0.99
    # LDA projects to C-1 components
    LDA_COMPONENTS = N_CLASSES - 1  # 98

    # Meta-Learner Input Dimension
    # 3 streams (DINO, ConvNeXt, Tabular) * 98 components each
    META_INPUT_DIM = 3 * LDA_COMPONENTS  # 294

    @classmethod
    def setup(cls):
        """
        Ensures all necessary working directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup()
