import os
import random
import numpy as np
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "BirdSpeciesClassification_ResNet18_SPP"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    ESSENTIAL_DATA_DIR = os.path.join(INPUT_DIR, "essential_data")

    # Working directory for caching intermediate files (spectrograms, etc.)
    # Strictly following the requirement to use ./working/idea_6/
    WORKING_DIR = "./working/idea_6/"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output submission file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Audio Processing Parameters
    # =========================================================================
    SR = 16000  # Sampling Rate (16kHz as per dataset)
    DURATION = 10  # Duration in seconds
    N_MELS = 128  # Number of Mel bands
    N_FFT = 1024  # FFT window size
    HOP_LENGTH = 320  # Hop length (approx 20ms)
    FMIN = 0  # Min frequency
    FMAX = 8000  # Max frequency (Nyquist)

    # =========================================================================
    # Model Parameters
    # =========================================================================
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 19
    IMAGE_SIZE = 224  # Input size for the model (224x224)
    IN_CHANNELS = 3  # Replicate mono spectrogram to 3 channels

    # SPP Grid levels (e.g., 1x1, 2x2, 4x4)
    SPP_LEVELS = [1, 2, 4]

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4  # Reduced from 1e-3 to improve stability
    WEIGHT_DECAY = 1e-4

    # Scheduler
    T_MAX = 10  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 10  # Epochs to wait before stopping

    # Mixup
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4

    # Cross Validation
    N_FOLDS = 5  # Number of folds for Iterative Stratification

    # =========================================================================
    # Setup Methods
    # =========================================================================
    @classmethod
    def setup(cls):
        """Creates necessary directories and sets random seeds."""
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Initialize setup immediately upon import to ensure directories exist
Config.setup()
