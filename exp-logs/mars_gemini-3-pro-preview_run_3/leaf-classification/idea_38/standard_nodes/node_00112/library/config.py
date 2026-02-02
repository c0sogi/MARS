import os
import torch


class Config:
    """
    Configuration class for the Leaf Classification project.
    Centralizes all file paths, hyperparameters, and model settings.
    """

    # ==========================================
    # 1. Directories & File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_38"
    SUBMISSION_DIR = "./submission"

    # Input Data
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Data
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Caching Paths (Deterministic Processing)
    # ==========================================
    # These paths are used to store/load extracted features to save time
    # and avoid re-computing heavy neural network forward passes.
    CACHE_TRAIN_IMG_FEATURES = os.path.join(WORKING_DIR, "train_img_features.npy")
    CACHE_TRAIN_TAB_FEATURES = os.path.join(WORKING_DIR, "train_tab_features.npy")
    CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "train_ids.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")

    CACHE_VAL_IMG_FEATURES = os.path.join(WORKING_DIR, "val_img_features.npy")
    CACHE_VAL_TAB_FEATURES = os.path.join(WORKING_DIR, "val_tab_features.npy")
    CACHE_VAL_IDS = os.path.join(WORKING_DIR, "val_ids.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")

    CACHE_TEST_IMG_FEATURES = os.path.join(WORKING_DIR, "test_img_features.npy")
    CACHE_TEST_TAB_FEATURES = os.path.join(WORKING_DIR, "test_tab_features.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

    # ==========================================
    # 3. Global Settings & Reproducibility
    # ==========================================
    SEED = 42
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50  # Number of samples to use if DEBUG is True

    # ==========================================
    # 4. Data Processing & Augmentation
    # ==========================================
    # 12 equidistant rotations: 0, 30, ..., 330
    ROTATION_ANGLES = [i * 30 for i in range(12)]

    # Image Parameters
    IMAGE_SIZE = 224
    # ImageNet Normalization
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    # ==========================================
    # 5. Model Architecture & Compute
    # ==========================================
    # Feature Extraction Models (timm)
    # DINOv2 Large for Global Geometry
    MODEL_DINO_NAME = "vit_large_patch14_dinov2.lvd142m"
    # ConvNeXt Large for Local Texture
    MODEL_CONVNEXT_NAME = "convnext_large.fb_in22k_ft_in1k"

    # Compute Settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    NUM_WORKERS = 4  # Adjust based on CPU cores (12 available)

    # ==========================================
    # 6. Pipeline Hyperparameters
    # ==========================================
    # Dimensionality Reduction
    PCA_VARIANCE = 0.99

    # Cross Validation
    N_FOLDS = 5

    # Linear Discriminant Analysis (LDA)
    # Note: LDA is a closed-form solution, so 'epochs' are not applicable.
    LDA_SOLVER = "lsqr"
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

    # Tabular Feature Transformation
    TABULAR_OUTPUT_DIST = "normal"  # QuantileTransformer output
