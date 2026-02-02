import os
import torch


class Config:
    """
    Configuration class for the Apple Disease Detection pipeline.
    Implements the 'Full-Dataset ResNet34 with Hybrid-Schedule SWA' strategy.
    """

    # ==========================================
    # General Setup
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on available vCPUs

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input
    INPUT_DIR = "./input"
    IMAGES_DIR = os.path.join(INPUT_DIR, "images")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output
    # Specific working directory for this idea to support caching requirements
    WORKING_DIR = "./working/idea_9"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create output directories immediately
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Data Configuration
    # ==========================================
    IMG_SIZE = 256
    NUM_CLASSES = 4
    CLASS_LABELS = ["healthy", "multiple_diseases", "rust", "scab"]

    # Debugging / Quick Run
    # If True, dataset classes should subsample the data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 0.1  # Fraction of data to use if DEBUG is True

    # ==========================================
    # Model Configuration
    # ==========================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True

    # ==========================================
    # Training Configuration
    # ==========================================
    BATCH_SIZE = 32

    # Training Configuration
    # Standard Schedule with Best Checkpointing (Cite Lesson 00031)
    EPOCHS = 15
    LR_START = 1e-4
