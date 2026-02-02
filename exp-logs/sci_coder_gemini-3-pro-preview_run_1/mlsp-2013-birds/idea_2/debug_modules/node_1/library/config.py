import os
import random
import numpy as np
import torch


class Config:
    """
    Central configuration for the Bird Species Classification task.
    """

    # ==========================================
    # Directories and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths
    SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    PREDICTIONS_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    NUM_CLASSES = 19
    # Original spectrograms are ~1246x256. We resize to a standard aspect ratio
    # compatible with CNNs while preserving some temporal resolution.
    IMG_HEIGHT = 256
    IMG_WIDTH = 512
    IN_CHANNELS = 3  # ResNet expects 3 channels; we will replicate the grayscale input

    # Debugging / Development
    DEBUG = False
    # If set to an integer, limits the dataset size for quick debugging cycles
    MAX_SAMPLES = None

    # ==========================================
    # Model Parameters
    # ==========================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True
    DROPOUT_RATE = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 8

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup_reproducibility(seed=42):
        """
        Sets the random seed for all relevant libraries to ensure reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
