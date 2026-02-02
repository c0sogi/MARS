import os
import torch


class Config:
    """
    Configuration for High-Fidelity Deep-Aligned Contextualized-Residual Network (HiFi-DACR).
    Centralizes all hyperparameters, paths, and constants.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    PROJECT_NAME = "idea_40"
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Directories & Paths
    # -------------------------------------------------------------------------
    ROOT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific output directory for this idea (used for checkpoints and cache)
    OUTPUT_DIR = os.path.join(WORKING_DIR, PROJECT_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Cache settings
    CACHE_DIR = OUTPUT_DIR
    USE_CACHE = True

    # Metadata file paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Model artifacts
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Preprocessing
    # -------------------------------------------------------------------------
    # Image resolution matches EfficientNet-B1 native resolution
    IMG_SIZE = 240

    # Tri-Slab Strategy
    NUM_SLABS = 3
    SLAB_OVERLAP = 0.15

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b1"
    PRETRAINED = True
    IN_CHANNELS = 3  # RGB (MIPs)

    # Feature Dimensions
    # EfficientNet-B1 outputs 1280-dim features at the final conv layer
    VISUAL_DIM = 1280

    # Tabular features are projected to match visual dimension for attention
    TABULAR_DIM = 1280

    # Tabular Input Features
    # 'Percent' is a critical prior; Age/Sex/Smoking are demographic context
    NUMERICAL_COLS = ["Age", "Percent"]
    CATEGORICAL_COLS = ["Sex", "SmokingStatus"]

    # Attention Mechanism (Pre-Norm Symmetric Attention)
    NUM_ATTENTION_HEADS = 4  # 1280 / 4 = 320 dim per head
    ATTENTION_LAYERS = 1
    FFN_DIM = 2048
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Debug flag to run on a small subset of data for quick verification
    DEBUG = False

    # Optimization
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Early Stopping
    PATIENCE = 8  # Strict patience to prevent overfitting on small dataset

    # -------------------------------------------------------------------------
    # Metric & Inference
    # -------------------------------------------------------------------------
    # Modified Laplace Log Likelihood constants
    SIGMA_CLIP = 70.0
    MAX_DELTA = 1000.0
