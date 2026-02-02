import os
import torch


class Config:
    # =========================================================================
    # Global Settings
    # =========================================================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # =========================================================================
    # File Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Cache directory for this specific idea/experiment
    # Using 'idea_7' as the identifier for this run
    WORKING_DIR = "./working/idea_7"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output submission path
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Image Configuration
    # =========================================================================
    IMG_SIZE = 224

    # Normalization constants
    # CLIP uses specific mean/std
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    # ImageNet standard (for DINOv2 and ConvNeXt)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    # =========================================================================
    # Model Backbones (Feature Extractors)
    # =========================================================================
    # Using HuggingFace Transformers / timm naming conventions
    BACKBONES = {
        "clip": {
            "name": "openai/clip-vit-large-patch14",
            "type": "clip",
            "mean": CLIP_MEAN,
            "std": CLIP_STD,
        },
        "dinov2": {
            "name": "facebook/dinov2-large",
            "type": "vit",
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "convnext": {
            "name": "facebook/convnext-large-224-22k-1k",
            "type": "cnn",
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
    }

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    N_FOLDS = 5
    BATCH_SIZE = 32

    # Level-0 Expert Hyperparameters

    # PCA for Tree-based models
    PCA_COMPONENTS = 64

    # Ridge Regression
    # Expanded alpha search range to prevent boundary clipping
    RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 50000.0]

    # SVR (Support Vector Regression)
    SVR_C = 1.0
    SVR_EPSILON = 0.1
    SVR_CACHE_SIZE = 2000  # MB

    # ExtraTrees
    ET_N_ESTIMATORS = 500
    ET_MAX_DEPTH = None
    ET_MIN_SAMPLES_SPLIT = 2
    ET_JOBS = -1

    # Level-1 Meta-Learner Hyperparameters
    # Bayesian Ridge is parameter-free regarding regularization (it infers it)
    META_MODEL_ITER = 300

    # =========================================================================
    # Feature Engineering
    # =========================================================================
    # List of binary metadata columns to include
    BINARY_FEATURES = [
        "Focus",
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
