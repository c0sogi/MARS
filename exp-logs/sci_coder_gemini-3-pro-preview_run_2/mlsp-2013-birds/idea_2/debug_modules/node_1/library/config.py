import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Bird Species Classification task.
    Centralizes all file paths, model hyperparameters, and training settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to use a small subset of data for debugging
    DEBUG_SUBSET_SIZE = 20
    NUM_WORKERS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Image Source
    # Using filtered spectrograms as proposed in the solution idea
    FILTERED_SPECTROGRAM_DIR = os.path.join(
        INPUT_ROOT, "supplemental_data", "filtered_spectrograms"
    )

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 19
    IN_CHANNELS = 3  # ResNet expects 3 channels; we will convert grayscale to RGB
    DROPOUT_RATE = 0.2

    # ==========================================
    # Data Preprocessing & Augmentation
    # ==========================================
    IMG_SIZE = (224, 224)
    # Standard ImageNet normalization statistics
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler Settings
    T_MAX = 30  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 7

    # Loss Function Settings
    # Placeholder for positive weights (can be calculated dynamically in training script)
    USE_POS_WEIGHT = True

    @classmethod
    def setup(cls):
        """
        Initialize directories and set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
