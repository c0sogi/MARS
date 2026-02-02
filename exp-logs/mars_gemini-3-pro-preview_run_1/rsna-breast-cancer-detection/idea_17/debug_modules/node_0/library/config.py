import os
import torch


class Config:
    """
    Global configuration for the Pyramid Symmetry-Difference Siamese Network.
    """

    # =========================================================================
    # Random Seed for Reproducibility
    # =========================================================================
    SEED = 42

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directories
    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_17")

    # Cache / Output Paths
    CACHE_DIR = IDEA_DIR
    MODEL_PATH = os.path.join(IDEA_DIR, "model.pth")
    AGE_STATS_PATH = os.path.join(IDEA_DIR, "age_stats.npy")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Input Dimensions: (Channels, Height, Width)
    # Channels = 3 (Image, Age Map, Implant Map)
    IMG_SIZE = (768, 768)
    CHANNELS = 3

    # Data Loading
    NUM_WORKERS = 4

    # =========================================================================
    # Model & Training Hyperparameters
    # =========================================================================
    # Architecture
    BACKBONE = "efficientnet_b2"

    # Training Loop
    BATCH_SIZE = 8  # Adjusted for 768x768 Siamese Network on A100
    EPOCHS = 5  # Number of training epochs

    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Loss Function (BCEWithLogitsLoss)
    # Aggressive weighting for high imbalance (approx 1:47 ratio)
    POS_WEIGHT = 47.0

    # Gradient Strategy
    CLIP_GRADIENTS = False  # Disabled to allow large updates for minority class

    # =========================================================================
    # Hardware & Debugging
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flags to control dataset size
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use if DEBUG is True

    @staticmethod
    def setup():
        """
        Ensures that necessary working directories exist.
        Should be called at the start of execution.
        """
        os.makedirs(Config.IDEA_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories when config is imported
Config.setup()
