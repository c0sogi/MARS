import os
import numpy as np


class Config:
    """
    Global configuration for the Selective-Topology Orthogonal Manifold-Densified LDA solution.
    """

    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Working directory for caching intermediate features (idea_26 specific)
    WORKING_DIR = "./working/idea_26"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # 2. Model Hyperparameters
    # ==========================================
    # Feature Extractors (timm compatible names)
    # DINOv2 Large for Global Geometry
    MODEL_DINO_NAME = "vit_large_patch14_dinov2.lvd142m"
    # ConvNeXt Large for Local Texture
    MODEL_CONVNEXT_NAME = "convnext_large.fb_in22k_ft_in1k"

    # Image Input Specs
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # ==========================================
    # 3. Data Processing (Rotations & Topology)
    # ==========================================
    # 12 Equidistant Rotations: 0, 30, 60, ..., 330
    NUM_ROTATIONS = 12
    ROTATION_ANGLES = np.linspace(0, 330, NUM_ROTATIONS).astype(int).tolist()

    # Orthogonal Centroid Grouping (Indices into the ROTATION_ANGLES list)
    # Centroid A: 0°, 90°, 180°, 270° -> Indices [0, 3, 6, 9]
    # Centroid B: 30°, 120°, 210°, 300° -> Indices [1, 4, 7, 10]
    # Centroid C: 60°, 150°, 240°, 330° -> Indices [2, 5, 8, 11]
    CENTROID_INDICES = {"A": [0, 3, 6, 9], "B": [1, 4, 7, 10], "C": [2, 5, 8, 11]}

    # ==========================================
    # 4. Feature Engineering & Training
    # ==========================================
    # Independent Subspace Reduction
    PCA_VARIANCE_THRESHOLD = 0.99

    # Cross-Validation
    N_FOLDS = 10
    RANDOM_SEED = 42

    # Debugging / Development
    # Set to an integer (e.g., 100) to limit dataset size during testing, None for full run
    DEBUG_SAMPLE_LIMIT = None

    # ==========================================
    # 5. Cache File Paths
    # ==========================================
    # These paths are used to store/load the extracted features to save time
    CACHE_PATH_TRAIN_FEATURES = os.path.join(
        WORKING_DIR, "train_features_densified.parquet"
    )
    CACHE_PATH_TEST_FEATURES = os.path.join(
        WORKING_DIR, "test_features_densified.parquet"
    )
    CACHE_PATH_MODELS = os.path.join(WORKING_DIR, "models")

    os.makedirs(CACHE_PATH_MODELS, exist_ok=True)
