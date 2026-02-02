import os
import torch


class Config:
    """
    Configuration class for the Catheter Detection task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    PROJECT_NAME = "catheter_detection_resnet"
    IDEA_NAME = "idea_1"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use if DEBUG is True

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata directories (Read-Only, Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output directories (Write access allowed)
    WORKING_DIR = "./working"
    OUTPUT_DIR = os.path.join(WORKING_DIR, IDEA_NAME)
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")

    # -------------------------------------------------------------------------
    # Data Hyperparameters
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 512
    IN_CHANNELS = 3  # Replicating single-channel X-ray to 3 channels for ResNet
    NUM_WORKERS = 12  # Using available vCPUs

    # Target Labels (11 classes)
    TARGET_COLS = [
        "ETT - Abnormal",
        "ETT - Borderline",
        "ETT - Normal",
        "NGT - Abnormal",
        "NGT - Borderline",
        "NGT - Incompletely Imaged",
        "NGT - Normal",
        "CVC - Abnormal",
        "CVC - Borderline",
        "CVC - Normal",
        "Swan Ganz Catheter Present",
    ]
    NUM_CLASSES = len(TARGET_COLS)

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    MODEL_NAME = "resnet34"
    PRETRAINED = True
    DROPOUT_RATE = (
        0.0  # ResNet usually doesn't need heavy dropout in the head, but can be added
    )

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0
    PATIENCE = 3  # Early stopping patience

    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def create_output_dirs(cls):
        """
        Creates necessary directories for outputs and submissions.
        """
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
