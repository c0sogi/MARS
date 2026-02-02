import os

# Ensure necessary directories exist upon module import
WORKING_DIR = "./working/idea_35"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


class Config:
    """
    Configuration class for the Leaf Classification Task.
    Implements settings for:
    - Stratified Selective-Topology Orthogonal Manifold-Densified LDA
    - Dual-Stream Feature Extraction (DINOv2 + ConvNeXt)
    - Data Paths and Caching
    """

    # ==========================================
    # Global & Debug Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Flag to run in debug mode with fewer samples
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use when DEBUG is True

    # ==========================================
    # File System Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working Directory for Caching Intermediate Features
    WORKING_DIR = WORKING_DIR

    # Submission Directory
    SUBMISSION_DIR = SUBMISSION_DIR

    # Metadata Files (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Final Output
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Feature Extraction Models & Parameters
    # ==========================================
    # Model Names (timm compatible)
    # Global Geometry Stream: DINOv2 Large
    MODEL_DINO = "vit_large_patch14_dinov2.lvd142m"

    # Local Texture Stream: ConvNeXt Large
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"

    # Image Parameters
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = "cuda"  # Assumes NVIDIA A100 availability

    # ==========================================
    # Manifold Densification (Rotations & Centroids)
    # ==========================================
    # 12 Equidistant Rotations (0° to 330° with 30° step)
    ROTATION_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

    # Centroid Definitions: Indices into ROTATION_ANGLES
    # Centroid A: {0°, 90°, 180°, 270°} -> Indices [0, 3, 6, 9]
    CENTROID_A_INDICES = [0, 3, 6, 9]

    # Centroid B: {30°, 120°, 210°, 300°} -> Indices [1, 4, 7, 10]
    CENTROID_B_INDICES = [1, 4, 7, 10]

    # Centroid C: {60°, 150°, 240°, 330°} -> Indices [2, 5, 8, 11]
    CENTROID_C_INDICES = [2, 5, 8, 11]

    # Mapping for easy access
    CENTROIDS = {
        "A": CENTROID_A_INDICES,
        "B": CENTROID_B_INDICES,
        "C": CENTROID_C_INDICES,
    }

    # ==========================================
    # Pipeline Hyperparameters
    # ==========================================
    # Independent Subspace Reduction
    PCA_VARIANCE = 0.99

    # Cross-Validation
    N_FOLDS = 10

    # Tabular Data Configuration
    # 3 sets of 64 features each = 192 total features
    TABULAR_PREFIXES = ["margin", "shape", "texture"]
    TABULAR_FEATURE_COUNT = 192

    # ==========================================
    # Cache File Names
    # ==========================================
    # Used to store extracted features to disk
    CACHE_TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")
    CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "train_ids.npy")

    CACHE_VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")
    CACHE_VAL_IDS = os.path.join(WORKING_DIR, "val_ids.npy")

    CACHE_TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")
