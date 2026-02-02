import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific working directory for this idea (Idea 26)
    IDEA_NAME = "idea_26"
    IDEA_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
    CACHE_DIR = IDEA_DIR

    # Submission Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Raw Data Files
    TRAIN_JSON_PATH = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON_PATH = os.path.join(INPUT_DIR, "test.json")

    # -------------------------------------------------------------------------
    # Data Processing Specifications
    # -------------------------------------------------------------------------
    IMG_HEIGHT = 75
    IMG_WIDTH = 75
    # Input Channels: 3 (HH, HV, and synthetic Average)
    IN_CHANNELS = 3

    # -------------------------------------------------------------------------
    # Model Hyperparameters (MAD-ResNet)
    # -------------------------------------------------------------------------
    # Custom 4-Stage Residual Network widths
    CHANNEL_WIDTHS = [64, 128, 128, 128]

    # Activation and Regularization
    LEAKY_RELU_SLOPE = 0.1
    DROPOUT_RATE = 0.2
    USE_BIAS = True  # Explicitly retain bias in Convolutions

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_FOLDS = 5
    BATCH_SIZE = 32
    NUM_EPOCHS = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization

    # -------------------------------------------------------------------------
    # Hardware Settings
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # -------------------------------------------------------------------------
    # Debugging / Control
    # -------------------------------------------------------------------------
    # Set DEBUG to True to run on a small subset of data for testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 100

    @classmethod
    def setup(cls):
        """
        Ensures necessary working and output directories exist.
        """
        os.makedirs(cls.IDEA_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup()
