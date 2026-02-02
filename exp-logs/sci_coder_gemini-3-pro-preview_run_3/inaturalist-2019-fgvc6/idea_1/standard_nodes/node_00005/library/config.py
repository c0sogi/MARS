import os
import torch


class Config:
    # =========================================================================
    # Directories and Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Specific Cache Directory for Idea 1
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_CHECKPOINT = os.path.join(CACHE_DIR, "best_model.pth")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    NUM_CLASSES = 1010

    # Image Preprocessing
    IMAGE_SIZE = 224  # Input size for the model
    RESIZE_SIZE = 256  # Initial resize before cropping

    # Data Loading
    BATCH_SIZE = 64  # Batch size for training
    VAL_BATCH_SIZE = 128  # Batch size for validation/inference
    NUM_WORKERS = 12  # Number of CPU workers for data loading

    # Debugging
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use in debug mode

    # =========================================================================
    # Model Configuration
    # =========================================================================
    MODEL_NAME = "tf_efficientnetv2_s.in1k"  # Timm model name
    PRETRAINED = True

    # Regularization
    DROP_RATE = 0.2  # Dropout rate for the classification head
    DROP_PATH_RATE = 0.1  # Stochastic depth rate
    LABEL_SMOOTHING = 0.1  # Label smoothing factor for CrossEntropy

    # =========================================================================
    # Training Configuration
    # =========================================================================
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW

    # Loss Configuration
    FOCAL_GAMMA = 2.0

    # Scheduler
    WARMUP_EPOCHS = 1
    MIN_LR = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 3

    # Reproducibility
    SEED = 42

    # =========================================================================
    # Hardware Configuration
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True  # Automatic Mixed Precision

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
