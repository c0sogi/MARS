import os
import torch


class Config:
    """
    Configuration module for Hyper-Densified Dual-Stream LDA with Full-Manifold Test-Time Aggregation.
    """

    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of workers for DataLoaders

    # ==========================================
    # Data Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # Output & Caching Directories
    # ==========================================
    # Working directory for this specific idea (Idea 19)
    WORKING_DIR = "./working/idea_19"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache File Paths (using .npy for efficiency)
    CACHE_CLASSES = os.path.join(WORKING_DIR, "classes.npy")

    # Train Caches
    CACHE_TRAIN_IMG_FEATURES = os.path.join(WORKING_DIR, "train_img_features.npy")
    CACHE_TRAIN_TAB_FEATURES = os.path.join(WORKING_DIR, "train_tab_features.npy")
    CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "train_ids.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    # Test Caches
    CACHE_TEST_IMG_FEATURES = os.path.join(WORKING_DIR, "test_img_features.npy")
    CACHE_TEST_TAB_FEATURES = os.path.join(WORKING_DIR, "test_tab_features.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # ==========================================
    # Feature Extraction Parameters
    # ==========================================
    # Image Processing
    IMAGE_SIZE = 224
    BATCH_SIZE = 32

    # Manifold Densification Topology
    NUM_ROTATIONS = (
        36  # Extract features for 36 equidistant rotations (0, 10, ..., 350)
    )
    NUM_CENTROIDS = 9  # Generate 9 centroids by averaging 4 orthogonal views

    # Model Architectures (timm compatible names)
    # DINOv2 Large: Captures global geometry
    MODEL_DINOV2 = "vit_large_patch14_dinov2.lvd142m"
    # ConvNeXt Large: Captures local texture/margin details
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"

    # Tabular Features
    TABULAR_PREFIXES = ["margin", "shape", "texture"]

    # ==========================================
    # Training & Evaluation Parameters
    # ==========================================
    N_FOLDS = 10
    PCA_VARIANCE = 0.99  # Variance retention threshold for PCA

    # Metric & Submission
    EPSILON = 1e-15
    CLIP_MIN = EPSILON
    CLIP_MAX = 1.0 - EPSILON
