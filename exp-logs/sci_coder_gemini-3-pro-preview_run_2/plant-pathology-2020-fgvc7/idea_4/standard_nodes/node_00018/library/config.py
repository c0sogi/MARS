import os
import torch


class Config:
    """
    Configuration class for Apple Disease Detection.
    Implements the strategy based on EfficientNet-B5 with Compound Scaling.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SUBSET_SIZE = 100

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")
    WORKING_DIR = "./working"

    # Caching directory for Idea 4 (Parquet/NPY files)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_4")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # EfficientNet-B5 Noisy Student (timm)
    MODEL_NAME = "tf_efficientnet_b5_ns"
    NUM_CLASSES = 4

    # Classification Head
    DROPOUT_RATE = 0.4  # Dropout before final linear layer

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    # Native resolution for EfficientNet-B5
    IMAGE_SIZE = 456

    # Batch size (16 is stable for B5 on 40GB GPU)
    BATCH_SIZE = 8

    # DataLoader workers
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Loss Function: Label Smoothing to prevent overconfidence
    LABEL_SMOOTHING = 0.1

    # Scheduler: Cosine Annealing
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # -------------------------------------------------------------------------
    # Augmentation (Albumentations)
    # -------------------------------------------------------------------------
    # CoarseDropout: Replaces CutMix for localized pathology detection.
    # Scaled for 456x456 resolution.
    COARSE_DROPOUT_PARAMS = {
        "max_holes": 8,
        "max_height": 100,
        "max_width": 100,
        "min_holes": 1,
        "min_height": 16,
        "min_width": 16,
        "fill_value": 0,
        "p": 0.5,
    }

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    # Test Time Augmentation (Horizontal Flip)
    TTA_FLIP = True

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for caching and submission.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup to ensure directories exist when config is imported
Config.setup()
