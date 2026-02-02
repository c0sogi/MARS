import os
import numpy as np

# Ensure necessary directories exist
os.makedirs("./working/idea_23", exist_ok=True)
os.makedirs("./submission", exist_ok=True)


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_23"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Model Configuration
    # ==========================================
    # Global Geometry Stream Model
    MODEL_DINO = "facebook/dinov2-large"

    # Local Texture Stream Model
    MODEL_CONVNEXT = "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384"

    # Input Image Parameters
    IMG_SIZE = 224
    BATCH_SIZE = 4

    # ==========================================
    # Algorithm Hyperparameters
    # ==========================================
    # Reproducibility
    SEED = 42

    # Validation Strategy
    N_FOLDS = 10

    # Independent Subspace Reduction
    PCA_VARIANCE = 0.99

    # ==========================================
    # Manifold Densification Configuration
    # ==========================================
    # 12 Equidistant rotations (0, 30, 60, ..., 330)
    NUM_ROTATIONS = 12
    ROTATION_ANGLES = [i * 30 for i in range(12)]

    # Orthogonal Centroid Groups (Indices corresponding to ROTATION_ANGLES)
    # Group A: {0, 90, 180, 270} -> Indices [0, 3, 6, 9]
    # Group B: {30, 120, 210, 300} -> Indices [1, 4, 7, 10]
    # Group C: {60, 150, 240, 330} -> Indices [2, 5, 8, 11]
    CENTROID_INDICES = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]

    # Feature Columns
    TABULAR_PREFIXES = ["margin", "shape", "texture"]

    # ==========================================
    # Caching Configuration
    # ==========================================
    # Paths for caching extracted features to speed up iterative development
    CACHE_TRAIN_IMG_FEATURES = os.path.join(WORKING_DIR, "train_img_features.npy")
    CACHE_TRAIN_TAB_FEATURES = os.path.join(WORKING_DIR, "train_tab_features.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")
    CACHE_TRAIN_IDS = os.path.join(WORKING_DIR, "train_ids.npy")

    CACHE_TEST_IMG_FEATURES = os.path.join(WORKING_DIR, "test_img_features.npy")
    CACHE_TEST_TAB_FEATURES = os.path.join(WORKING_DIR, "test_tab_features.npy")
    CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")
