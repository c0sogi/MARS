import os
import numpy as np


class Config:
    """
    Global configuration for the Pawpularity Prediction Pipeline.
    Implements the 'Quad-Paradigm Stacking Ensemble with Zero-Shot Aesthetic Injection'.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEBUG_SAMPLE_SIZE = 100

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate features (Idea 8)
    WORKING_DIR = "./working/idea_8"

    # Output directory for submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Model Backbones (Feature Extractors)
    # =========================================================================
    # 1. Semantic Expert: CLIP ViT-L/14@336px
    # Using the OpenAI CLIP weights compatible with transformers/sentence-transformers
    MODEL_CLIP = "openai/clip-vit-large-patch14-336"
    IMG_SIZE_CLIP = 336

    # 2. Geometric Expert: DINOv2 ViT-Large
    # timm model identifier
    MODEL_DINO = "vit_large_patch14_dinov2.lvd142m"
    IMG_SIZE_DINO = 224

    # 3. Textural Expert: ConvNeXt Large
    # timm model identifier
    MODEL_CONVNEXT = "convnext_large.fb_in22k_ft_in1k"
    IMG_SIZE_CONVNEXT = 224

    # =========================================================================
    # Feature Engineering
    # =========================================================================
    # Aesthetic Prompts for Zero-Shot Aesthetic Injection
    # These prompts are used to compute cosine similarity scores with image embeddings
    AESTHETIC_PROMPTS = [
        "a cute pet",
        "an adorable photo",
        "a high quality image",
        "a blurry photo",
        "a funny pet",
        "a sad looking pet",
        "a happy pet",
        "a scary pet",
        "well focused",
        "bright image",
        "dark image",
        "cluttered background",
        "clean background",
    ]

    # Dimensionality Reduction for Tree/KNN based models
    PCA_COMPONENTS = 64

    # =========================================================================
    # Training & Validation
    # =========================================================================
    N_FOLDS = 5

    # =========================================================================
    # Level-0 Expert Hyperparameters
    # =========================================================================

    # 1. Ridge Regression (Linear Expert)
    # Search range for regularization strength (alpha)
    # Log-spaced from 0.001 to 50,000 to cover wide regularization scales
    RIDGE_ALPHAS = np.logspace(-3, np.log10(50000), 100).tolist()

    # 2. K-Nearest Neighbors (Retrieval Expert)
    # Number of neighbors to consider
    KNN_NEIGHBORS = [5, 10, 15, 20, 30, 40, 50]
    KNN_METRIC = "cosine"

    # 3. Support Vector Regression (Kernel Expert)
    SVR_C = 1.0
    SVR_KERNEL = "rbf"
    SVR_EPSILON = 0.1

    # 4. ExtraTrees Regressor (Partitioning Expert)
    ET_N_ESTIMATORS = 100
    ET_MAX_DEPTH = None  # Allow full depth
    ET_MIN_SAMPLES_SPLIT = 5
    ET_MIN_SAMPLES_LEAF = 2

    # =========================================================================
    # Level-1 Meta-Learner Hyperparameters
    # =========================================================================
    # Bayesian Ridge is used for the meta-learner to handle collinearity of experts
    META_N_ITER = 300
    META_TOL = 1e-3

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print("CONFIG")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k}: {v}")
        print("=" * 40)
