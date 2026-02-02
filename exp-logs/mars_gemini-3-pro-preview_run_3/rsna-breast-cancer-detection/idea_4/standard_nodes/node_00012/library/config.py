import os
import torch


class Config:
    """
    Central configuration for the Breast Cancer Detection pipeline.
    Handles paths, hyperparameters, model settings, and system configurations.
    """

    # =========================================================================
    # System & Reproducibility
    # =========================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use available vCPUs, leaving some overhead
    NUM_WORKERS = 8

    # =========================================================================
    # File Paths
    # =========================================================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
    TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

    # Metadata Paths (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (Writable Cache & Checkpoints)
    WORKING_DIR = "./working/idea_4"
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # =========================================================================
    # Data Configuration
    # =========================================================================
    # Image Dimensions (Height, Width)
    # Mammograms are typically tall; resizing to fixed size for batching.
    IMG_HEIGHT = 1024
    IMG_WIDTH = 512

    # Debugging / Quick Iteration
    # Set DEBUG = True to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500  # Number of patients to use in debug mode

    # =========================================================================
    # Model Architecture
    # =========================================================================
    BACKBONE_NAME = "efficientnet_v2_s"
    # The model expects a bag of images (views) per patient-laterality

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    # Batch size represents number of 'bags' (breasts) per step
    # Reduced from 8 to 4 to avoid OOM on 16GB GPU (Cite debug_lesson_3)
    BATCH_SIZE = 4

    # Accumulate gradients to maintain effective batch size of 16
    GRADIENT_ACCUMULATION_STEPS = 4

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4

    # Training duration
    NUM_EPOCHS = 20

    # Early Stopping parameters
    PATIENCE = 5

    # =========================================================================
    # Calibration / Analytical Correction
    # =========================================================================
    # We use balanced sampling during training (50% positive, 50% negative)
    TRAIN_PREVALENCE = 0.50

    # The approximate prevalence in the test set (and general screening population)
    TEST_PREVALENCE = 0.02

    @classmethod
    def setup(cls):
        """
        Initialize the environment by creating necessary writable directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Automatically setup directories on import
Config.setup()
