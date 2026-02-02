import os
import torch


class Config:
    """
    Configuration class for Apple Disease Detection task.
    Centralizes hyperparameters, file paths, and system settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    WORKING_DIR = "./working"
    # Directory for caching deterministic data processing (Idea 1 specific)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Data Configuration
    # ==========================================
    # Target Classes
    CLASSES = ["healthy", "multiple_diseases", "rust", "scab"]
    NUM_CLASSES = len(CLASSES)

    # Image Preprocessing
    IMG_SIZE = 224  # EfficientNet-B0 native resolution
    IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet Mean
    IMG_STD = [0.229, 0.224, 0.225]  # ImageNet Std

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    DROPOUT_RATE = 0.2

    # Training Settings
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 3  # Stop if validation metric doesn't improve for 3 epochs

    # ==========================================
    # Hardware & System
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for caching and submissions.
        """
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Execute setup on module import to ensure directories exist
Config.setup()
