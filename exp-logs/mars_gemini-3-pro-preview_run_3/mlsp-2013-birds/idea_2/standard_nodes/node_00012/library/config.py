import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    ESSENTIAL_DATA_DIR = os.path.join(INPUT_ROOT, "essential_data")
    AUDIO_DIR = os.path.join(ESSENTIAL_DATA_DIR, "src_wavs")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"
    OUTPUT_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Audio Processing
    # =========================================================================
    SR = 16000  # Sampling rate
    DURATION = 10  # Duration in seconds
    N_MELS = 128  # Number of Mel bands
    N_FFT = 1024  # FFT window size
    HOP_LENGTH = 320  # Hop length (approx 20ms)
    F_MIN = 20  # Minimum frequency
    F_MAX = 8000  # Maximum frequency (Nyquist)
    TOP_DB = 80  # Top decibel for conversion

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 19
    PRETRAINED = True
    IN_CHANNELS = 1  # Mono audio converted to 1-channel spectrogram

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    N_FOLDS = 5
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience

    # Scheduler settings (Cosine Annealing)
    T_MAX = 10  # For CosineAnnealingLR or T_0 for WarmRestarts
    ETA_MIN = 1e-6

    # =========================================================================
    # Augmentation
    # =========================================================================
    MIXUP_ALPHA = 0.2
    SPECAUG_FREQ_MASK_PARAM = 20
    SPECAUG_TIME_MASK_PARAM = 40
    NOISE_AMPLITUDE = 0.001
    GAIN_MIN = -3.0  # dB
    GAIN_MAX = 3.0  # dB

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Adjust based on vCPU count (12 available)


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
