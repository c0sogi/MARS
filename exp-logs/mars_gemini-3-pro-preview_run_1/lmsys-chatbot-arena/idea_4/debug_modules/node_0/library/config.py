import os
import torch


class Config:
    """
    Central configuration for the Siamese DeBERTa-v3-Base solution.
    Handles file paths, model architecture settings, and training hyperparameters.
    """

    # --------------------------------------------------------------------------
    # General & Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 500

    # Compute Environment
    # Automatically detect GPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of dataloader workers

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Metadata (Read-Only)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (Write Allowed)
    WORKING_DIR = "./working/idea_4"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 512
    NUM_CLASSES = 3  # Winner A, Winner B, Tie

    # Weighted Layer Pooling Settings
    # We use the weighted average of the last N hidden layers
    USE_WEIGHTED_LAYER_POOLING = True
    POOLING_LAYERS = 4

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 8
    EPOCHS = 4

    # Differential Learning Rates
    # Lower LR for the pre-trained backbone to preserve knowledge
    # Higher LR for the head/pooling layers to learn task specifics
    LR_BACKBONE = 1e-5
    LR_HEAD = 1e-3

    # Optimization
    WEIGHT_DECAY = 0.01
    EPS = 1e-6
    MAX_GRAD_NORM = 1.0

    # Scheduler
    NUM_WARMUP_STEPS_RATIO = 0.1

    # Early Stopping
    PATIENCE = 2

    # --------------------------------------------------------------------------
    # Feature Engineering & Caching
    # --------------------------------------------------------------------------
    # Versioning ensures we don't load stale cache if logic changes
    FEAT_VERSION = "v1"

    # Cache file paths
    TRAIN_FEATURES_PATH = os.path.join(CACHE_DIR, f"train_features_{FEAT_VERSION}.npy")
    VAL_FEATURES_PATH = os.path.join(CACHE_DIR, f"val_features_{FEAT_VERSION}.npy")
    TEST_FEATURES_PATH = os.path.join(CACHE_DIR, f"test_features_{FEAT_VERSION}.npy")

    @classmethod
    def create_dirs(cls):
        """
        Creates the necessary directory structure for outputs and cache.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically ensure directories exist when config is imported
Config.create_dirs()
