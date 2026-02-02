import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration for Right Whale Detection Task (Idea 10).
    Implements parameters for Time-Preserving CSK-ResNet-18 CRNN.
    """

    # ==========================================
    # Project & Paths
    # ==========================================
    PROJECT_NAME = "idea_10"
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and model checkpoints
    WORKING_DIR = f"./working/{PROJECT_NAME}"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Audio Processing
    # ==========================================
    SAMPLE_RATE = 2000
    DURATION = 2.0  # Seconds (Full clip duration, do not truncate)

    # Spectrogram Parameters
    # N_FFT=256 at 2000Hz -> ~128ms window
    # HOP_LENGTH=16 at 2000Hz -> ~8ms step.
    # For 2.0s (4000 samples), this yields 4000/16 = 250 time frames.
    # With Time-Preserving strides (total stride 8 in time), final seq len is ~31.
    N_MELS = 128
    N_FFT = 256
    HOP_LENGTH = 16
    F_MIN = 20
    F_MAX = 1000  # Nyquist frequency

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "CSKResNet18CRNN"
    NUM_CLASSES = 1
    IN_CHANNELS = 1  # Using 1-channel Log-Mel Spectrogram

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Class Imbalance Handling
    # Positive Class Weight for BCEWithLogitsLoss
    POS_WEIGHT = 9.0

    # Regularization
    MIXUP_ALPHA = 0.4

    # SpecAugment Parameters
    # Time Mask: Max 200ms.
    # With HOP_LENGTH=16 (8ms), 200ms approx 25 frames.
    TIME_MASK_PARAM = 25
    FREQ_MASK_PARAM = 20

    # ==========================================
    # Compute & Environment
    # ==========================================
    NUM_WORKERS = 2
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging & Caching
    # ==========================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500  # Number of samples to use when DEBUG=True
    CACHE_DIR = WORKING_DIR

    @classmethod
    def setup(cls):
        """
        Initialize the project environment.
        Creates necessary directories for working files and submissions.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def set_seed(seed=42):
        """
        Set random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Automatically setup directories on import
Config.setup()
