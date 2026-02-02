import os
import numpy as np


class Config:
    """
    Configuration for Orientation-Specialized Linear Discriminant Experts (OS-LDE).
    Centralizes paths, hyperparameters, and constants for the pipeline.
    """

    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_21"
    SUBMISSION_DIR = "./submission"

    # Ensure essential writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Data Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Output Paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Global Runtime Settings
    # ==========================================
    SEED = 42
    N_FOLDS = 5
    NUM_WORKERS = 4

    # Debugging / Development
    # Set DEBUG to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 50

    # ==========================================
    # 3. Data Processing & Augmentation
    # ==========================================
    IMG_SIZE = 224
    BATCH_SIZE = 32  # Batch size for feature extraction inference

    # Rotation Configuration
    # We extract features for 12 equidistant rotations
    NUM_ROTATIONS = 12
    # Angles: [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    ROTATION_ANGLES = [i * 30 for i in range(NUM_ROTATIONS)]

    # Orthogonal View-Set Partitioning
    # Maps the expert ID to the indices of ROTATION_ANGLES
    ORTHOGONAL_SETS = {
        # Expert 1: 0-Series (0, 90, 180, 270)
        "A": [0, 3, 6, 9],
        # Expert 2: 30-Series (30, 120, 210, 300)
        "B": [1, 4, 7, 10],
        # Expert 3: 60-Series (60, 150, 240, 330)
        "C": [2, 5, 8, 11],
    }

    # ==========================================
    # 4. Model Architecture & Hyperparameters
    # ==========================================
    # Feature Extractors (timm library names)
    # Global Geometry Stream
    MODEL_DINO = "vit_large_patch14_dinov2"
    # Local Texture Stream
    MODEL_CONVNEXT = "convnext_large"

    # Dimensionality Reduction
    # Retain 99% variance in PCA for each stream independently
    PCA_VARIANCE = 0.99

    # Classifier (Linear Discriminant Analysis)
    LDA_SOLVER = "lsqr"  # Least squares solution, supports shrinkage
    LDA_SHRINKAGE = "auto"  # Ledoit-Wolf shrinkage

    # Post-Processing
    PROB_CLIP_MIN = 1e-15
    PROB_CLIP_MAX = 1.0 - 1e-15

    # ==========================================
    # 5. Caching Naming Conventions
    # ==========================================
    @staticmethod
    def get_cache_path(filename):
        """Helper to generate full paths for cached files in working directory."""
        return os.path.join(Config.WORKING_DIR, filename)

    # Cache Keys
    CACHE_TRAIN_IMG_FEATURES = "train_img_features_12views.npy"
    CACHE_TEST_IMG_FEATURES = "test_img_features_12views.npy"
    CACHE_TRAIN_TABULAR = "train_tabular.npy"
    CACHE_TEST_TABULAR = "test_tabular.npy"
    CACHE_TRAIN_IDS = "train_ids.npy"
    CACHE_TEST_IDS = "test_ids.npy"
    CACHE_TRAIN_LABELS = "train_labels.npy"
    CACHE_CLASSES = "classes.npy"
