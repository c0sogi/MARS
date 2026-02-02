import os
import torch


class Config:
    # =========================================================================
    # Project & Paths
    # =========================================================================
    PROJECT_NAME = "idea_6"
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Output directory for checkpoints, cache, and predictions
    OUTPUT_DIR = os.path.join("./working", PROJECT_NAME)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Audio Directories
    TRAIN_CURATED_DIR = os.path.join(INPUT_ROOT, "train_curated")
    TRAIN_NOISY_DIR = os.path.join(INPUT_ROOT, "train_noisy")
    TEST_DIR = os.path.join(INPUT_ROOT, "test")

    # =========================================================================
    # Audio Parameters
    # =========================================================================
    SR = 32000  # Sampling Rate
    N_MELS = 128  # Number of Mel bins
    N_FFT = 1024  # FFT window size
    HOP_LENGTH = 320  # Hop length
    FMIN = 20  # Min frequency
    FMAX = 16000  # Max frequency
    DURATION = 5.0  # Duration of audio crops for training (seconds)

    # =========================================================================
    # Model Parameters
    # =========================================================================
    BACKBONE = "convnext_nano"  # Architecture
    PRETRAINED = True  # Use ImageNet weights
    NUM_CLASSES = 80  # Number of target classes
    IN_CHANNELS = 1  # Raw spectrogram channels (will be repeated to 3)

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    BATCH_SIZE = 64  # Optimized for A100 40GB
    EPOCHS = 30  # Extended schedule for ConvNeXt + Mixup
    LR = 1e-3  # Initial learning rate
    MIN_LR = 1e-6  # Minimum learning rate for scheduler
    WEIGHT_DECAY = 1e-2  # Weight decay for AdamW
    MIXUP_ALPHA = 0.4  # Strong regularization

    # =========================================================================
    # Hardware & System
    # =========================================================================
    NUM_WORKERS = 12  # Number of DataLoader workers (matches vCPUs)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Initialize the project environment.
        Creates necessary directories.
        """
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


# Automatically setup environment on import
Config.setup()
