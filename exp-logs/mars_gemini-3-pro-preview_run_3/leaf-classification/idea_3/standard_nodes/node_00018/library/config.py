import os
import torch


class Config:
    """
    Configuration class for the Leaf Classification pipeline.
    Defines paths, hyperparameters, and global constants.
    """

    # ==========================================
    # Path Configurations
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Reproducibility & Compute
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # Data Processing Parameters
    # ==========================================
    BATCH_SIZE = 32
    IMG_SIZE = 224

    # ImageNet Normalization Stats
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # Augmentation: Rotations for view averaging (Canonical Embedding)
    ROTATION_ANGLES = [0, 90, 180, 270]
    USE_FLIPS = True

    # Debugging: Set to an integer (e.g., 50) to limit dataset size
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Backbones
    BACKBONE_CNN = "convnext_tiny"
    BACKBONE_VIT = "vit_base_patch16_224"

    # Feature Engineering
    PCA_VARIANCE = 0.99  # Retain 99% variance

    # Classifiers
    # Logistic Regression
    LR_SOLVER = "lbfgs"
    LR_MAX_ITER = 2000
    LR_C = 1.0

    # Linear Discriminant Analysis
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

    # Ensemble
    ENSEMBLE_WEIGHT_STEP = 0.01

    # ==========================================
    # Caching Filenames (NPY format)
    # ==========================================
    # CNN Embeddings
    CACHE_TRAIN_CNN = "train_cnn_embeddings.npy"
    CACHE_VAL_CNN = "val_cnn_embeddings.npy"
    CACHE_TEST_CNN = "test_cnn_embeddings.npy"

    # ViT Embeddings
    CACHE_TRAIN_VIT = "train_vit_embeddings.npy"
    CACHE_VAL_VIT = "val_vit_embeddings.npy"
    CACHE_TEST_VIT = "test_vit_embeddings.npy"

    # Tabular Features (Processed)
    CACHE_TRAIN_TAB = "train_tabular.npy"
    CACHE_VAL_TAB = "val_tabular.npy"
    CACHE_TEST_TAB = "test_tabular.npy"

    # Targets
    CACHE_TRAIN_TARGETS = "train_targets.npy"
    CACHE_VAL_TARGETS = "val_targets.npy"

    # Test IDs (for submission)
    CACHE_TEST_IDS = "test_ids.npy"

    # Classes (Label Encoder)
    CACHE_CLASSES = "classes.npy"

    # PCA States (Mean and Components)
    CACHE_PCA_CNN_MEAN = "pca_cnn_mean.npy"
    CACHE_PCA_CNN_COMPONENTS = "pca_cnn_components.npy"
    CACHE_PCA_VIT_MEAN = "pca_vit_mean.npy"
    CACHE_PCA_VIT_COMPONENTS = "pca_vit_components.npy"

    # Scaler States (Mean and Scale)
    CACHE_SCALER_MEAN = "scaler_mean.npy"
    CACHE_SCALER_SCALE = "scaler_scale.npy"

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_cache_path(cls, filename):
        """
        Returns the full path for a cached file.
        """
        return os.path.join(cls.WORKING_DIR, filename)
