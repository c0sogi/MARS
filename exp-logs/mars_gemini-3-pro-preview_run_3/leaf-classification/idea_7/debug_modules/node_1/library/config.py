import os


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Dedicated cache directory for this experiment (Idea 7)
    IDEA_NAME = "idea_7"
    CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

    # Input Data Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architectures & Resolutions
    # ==========================================
    # Stream A: Global Geometry (DINOv2 ViT-Large)
    # Captures global shape and structure
    MODEL_DINO = "vit_large_patch14_dinov2.lvd142m"
    IMG_SIZE_DINO = 518

    # Stream B: Local Margin/Texture (ConvNeXt Large)
    # Captures high-frequency details (serrations, texture)
    # Using high resolution (1024) to preserve fine margin details
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"
    IMG_SIZE_CONVNEXT = 1024

    # Inference Parameters
    # Batch size is kept low due to large image size (1024x1024)
    BATCH_SIZE = 2

    # ==========================================
    # Feature Engineering
    # ==========================================
    # PCA Variance Retention
    PCA_VARIANCE = 0.99

    # Tabular Feature Prefixes
    TABULAR_PREFIXES = ["margin", "shape", "texture"]

    # ==========================================
    # Training / Ensemble Strategy
    # ==========================================
    # Cross-Validation
    N_FOLDS = 10  # 10-fold to maximize training data per fold

    # Classifier (LDA with Shrinkage)
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf

    # Post-processing
    PROB_CLIP_MIN = 1e-15
    PROB_CLIP_MAX = 1.0 - 1e-15

    # ==========================================
    # Caching Filenames (Numpy/Parquet)
    # ==========================================
    # Feature Caches
    CACHE_TRAIN_DINO = os.path.join(CACHE_DIR, "train_dino_features.npy")
    CACHE_TRAIN_CONV = os.path.join(CACHE_DIR, "train_conv_features.npy")
    CACHE_TEST_DINO = os.path.join(CACHE_DIR, "test_dino_features.npy")
    CACHE_TEST_CONV = os.path.join(CACHE_DIR, "test_conv_features.npy")

    # Tabular & Meta Caches
    CACHE_TRAIN_TABULAR = os.path.join(CACHE_DIR, "train_tabular.npy")
    CACHE_TEST_TABULAR = os.path.join(CACHE_DIR, "test_tabular.npy")
    CACHE_TRAIN_LABELS = os.path.join(CACHE_DIR, "train_labels.npy")
    CACHE_TEST_IDS = os.path.join(CACHE_DIR, "test_ids.npy")
    CACHE_CLASSES = os.path.join(CACHE_DIR, "classes.npy")

    # Pipeline Storage Pattern
    PIPELINE_PATH = os.path.join(CACHE_DIR, "pipeline_fold_{fold}.pkl")

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
