import os
import torch
import random
import numpy as np


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    # -------------------------------------------------------------------------
    # Directory and File Paths
    # -------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORK_DIR = "./working/idea_1"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # CSV Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Audio Processing Parameters
    # -------------------------------------------------------------------------
    SR = 32000  # Sampling rate
    DURATION = 30  # Duration of audio clips in seconds
    N_MELS = 128  # Number of Mel bands
    N_FFT = 1024  # FFT window size
    HOP_LENGTH = 320  # Hop length (approx 10ms at 32kHz)
    F_MIN = 0  # Minimum frequency
    F_MAX = None  # Maximum frequency (None -> SR/2)
    TOP_DB = 80  # Top decibels for signal normalization

    # -------------------------------------------------------------------------
    # Model Parameters
    # -------------------------------------------------------------------------
    MODEL_NAME = "mobilenet_v3_small"
    PRETRAINED = True
    NUM_CLASSES = 80
    IN_CHANNELS = 3  # Input channels (spectrogram replicated)

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32  # Adjusted for A100 GPU
    EPOCHS = 20  # Total training epochs
    LR = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-4  # Optimizer weight decay
    NUM_WORKERS = 12  # Number of DataLoader workers (using available vCPUs)

    # Scheduler settings
    T_MAX = EPOCHS  # For CosineAnnealingLR

    # -------------------------------------------------------------------------
    # Compute Settings
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 100  # Number of samples to use when DEBUG is True

    @classmethod
    def get_audio_params(cls):
        """Returns audio parameters as a dictionary."""
        return {
            "sr": cls.SR,
            "n_mels": cls.N_MELS,
            "n_fft": cls.N_FFT,
            "hop_length": cls.HOP_LENGTH,
            "f_min": cls.F_MIN,
            "f_max": cls.F_MAX,
            "top_db": cls.TOP_DB,
        }
