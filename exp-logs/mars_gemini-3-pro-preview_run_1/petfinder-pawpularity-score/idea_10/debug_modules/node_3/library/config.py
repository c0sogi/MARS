import os
import torch


class Config:
    # =========================================================================
    # General Configuration
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPUs (12 available)

    # =========================================================================
    # File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for this idea
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_10")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Dataset Information
    # =========================================================================
    ID_COL = "Id"
    TARGET_COL = "Pawpularity"
    FILE_PATH_COL = "file_path"

    # Binary metadata features provided in the dataset
    METADATA_COLS = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]

    # =========================================================================
    # Backbone Models (Feature Extractors)
    # =========================================================================
    # Defining the three experts with their native resolutions
    BACKBONES = {
        "siglip": {
            "model_name": "google/siglip-so400m-patch14-384",
            "resolution": 384,
            "batch_size": 32,
            "output_dim": 1152,  # SigLIP So400M embedding dimension
        },
        "dinov2": {
            "model_name": "facebook/dinov2-large",
            "resolution": 518,
            "batch_size": 16,  # Larger resolution requires smaller batch
            "output_dim": 1024,  # ViT-Large embedding dimension
        },
        "convnext": {
            "model_name": "facebook/convnext-large-224-22k-1k",
            "resolution": 224,
            "batch_size": 64,
            "output_dim": 1536,  # ConvNeXt Large embedding dimension
        },
    }

    # Feature-Space Augmentation
    USE_FLIP_AUGMENTATION = True

    # =========================================================================
    # Level-0 Experts (Heterogeneous Stack)
    # =========================================================================
    # 1. Ridge Regression Hyperparameters
    RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 50000.0]

    # 2. Support Vector Regression (SVR) Hyperparameters
    SVR_GRID = {
        "kernel": ["rbf"],
        "C": [0.1, 1.0, 10.0, 50.0],
        "epsilon": [0.01, 0.1, 0.5],
    }

    # 3. ExtraTrees Regressor Hyperparameters
    ET_PARAMS = {"n_estimators": 500, "n_jobs": -1, "random_state": SEED, "verbose": 0}
    # Grid for ExtraTrees tuning
    ET_GRID = {"max_depth": [None, 10, 20], "min_samples_leaf": [1, 4]}

    # Dimensionality Reduction for Tree-based models
    PCA_COMPONENTS = 64

    # =========================================================================
    # Level-1 Meta-Learner
    # =========================================================================
    META_MODEL_PARAMS = {"n_iter": 300, "verbose": False, "compute_score": True}

    # =========================================================================
    # Training Configuration
    # =========================================================================
    N_FOLDS = 5
