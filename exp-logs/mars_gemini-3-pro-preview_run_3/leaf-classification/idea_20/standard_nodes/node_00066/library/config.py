import os
import torch


class Config:
    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    N_FOLDS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Directory & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working directory for caching (idea_20)
    WORKING_DIR = "./working/idea_20"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    # Image parameters
    IMAGE_SIZE = 224

    # Manifold Densification (Rotation)
    NUM_ROTATIONS = 12  # 0, 30, 60, ..., 330 degrees
    ROTATION_STEP = 360 // NUM_ROTATIONS

    # Orthogonal Centroid Logic
    # We create 3 centroids, each averaging 4 orthogonal views
    NUM_CENTROIDS = 3
    VIEWS_PER_CENTROID = 4

    # Tabular Feature Prefixes
    TABULAR_PREFIXES = ["margin", "shape", "texture"]

    # ==========================================
    # Model Architectures & Feature Extraction
    # ==========================================
    # Model names compatible with timm
    # Using 'vit_large_patch14_dinov2' for global geometry
    # Using 'convnext_large' for local texture
    MODEL_DINO = "vit_large_patch14_dinov2.lvd142m"
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"

    # Feature Extraction Hyperparameters
    BATCH_SIZE = 4
    NUM_WORKERS = 4

    # ==========================================
    # Dimensionality Reduction & Classifier
    # ==========================================
    # PCA Variance Retention
    PCA_VARIANCE = 0.99

    # Classifier Settings (LDA)
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"

    # ==========================================
    # Caching Filenames
    # ==========================================
    # These files will be stored in WORKING_DIR
    # Using .npy for efficient storage of numpy arrays
    CACHE_FILE_MAP = {
        "train_img_features": "train_densified_img_features.npy",
        "train_tab_features": "train_densified_tab_features.npy",
        "train_labels": "train_densified_labels.npy",
        "train_ids": "train_densified_ids.npy",
        "test_img_features": "test_densified_img_features.npy",
        "test_tab_features": "test_densified_tab_features.npy",
        "test_ids": "test_densified_ids.npy",
    }

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_cache_path(cls, key):
        """Returns the full path for a cached file key."""
        filename = cls.CACHE_FILE_MAP.get(key)
        if filename is None:
            raise ValueError(f"Cache key '{key}' not found in configuration.")
        return os.path.join(cls.WORKING_DIR, filename)
