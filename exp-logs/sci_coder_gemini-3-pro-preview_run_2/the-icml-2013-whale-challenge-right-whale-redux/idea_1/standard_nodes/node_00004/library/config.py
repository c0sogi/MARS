import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio Preprocessing Parameters
    # ==========================================
    # Native sample rate is 2000Hz. Right whale calls are low freq (50-250Hz).
    SAMPLE_RATE = 2000
    DURATION = 2.0  # Fixed duration in seconds for padding/truncating

    # Spectrogram Parameters
    N_MELS = 128
    # N_FFT = 512 at 2000Hz gives ~3.9Hz freq resolution, good for low freq analysis
    N_FFT = 512
    HOP_LENGTH = 128
    F_MIN = 10  # Lower bound for Mel filterbank
    F_MAX = 1000  # Upper bound (Nyquist is 1000Hz)

    # Augmentation Parameters
    FREQ_MASK_PARAM = 20
    TIME_MASK_PARAM = 10

    # ==========================================
    # Model Parameters
    # ==========================================
    MODEL_NAME = "resnet18"
    IMG_SIZE = (224, 224)  # Standard input size for ResNet
    PRETRAINED = True
    NUM_CLASSES = 1
    DROPOUT = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 15
    PATIENCE = 4  # Early stopping patience
    SEED = 42

    # ==========================================
    # Debugging & Resources
    # ==========================================
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SAMPLES = 200

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Creates necessary working and submission directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def set_seed(cls):
        """
        Sets fixed random seeds for reproducibility across libraries.
        """
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
