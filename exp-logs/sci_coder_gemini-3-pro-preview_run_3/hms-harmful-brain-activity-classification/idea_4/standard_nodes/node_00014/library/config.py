import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for deterministic data processing)
    CACHE_TRAIN_DATA = os.path.join(WORKING_DIR, "train_data.npy")
    CACHE_TRAIN_TARGETS = os.path.join(WORKING_DIR, "train_targets.npy")
    CACHE_VAL_DATA = os.path.join(WORKING_DIR, "val_data.npy")
    CACHE_VAL_TARGETS = os.path.join(WORKING_DIR, "val_targets.npy")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    SEED = 42
    N_CLASSES = 6
    CLASS_NAMES = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    TARGET_COLS = [
        "seizure_prob",
        "lpd_prob",
        "gpd_prob",
        "lrda_prob",
        "grda_prob",
        "other_prob",
    ]

    # EEG Channel Configuration (Standard 10-20 System)
    # Excludes EKG if present in raw data
    EEG_CHANNELS = [
        "Fp1",
        "F3",
        "C3",
        "P3",
        "F7",
        "T3",
        "T5",
        "O1",
        "Fz",
        "Cz",
        "Pz",
        "Fp2",
        "F4",
        "C4",
        "P4",
        "F8",
        "T4",
        "T6",
        "O2",
    ]
    N_CHANNELS = 19

    # Stream A: Raw Waveform Parameters
    ORIGINAL_SAMPLING_RATE = 200  # Hz
    RAW_SAMPLING_RATE = 50  # Hz (Downsampled)
    DURATION = 50  # Seconds
    RAW_SEQUENCE_LENGTH = 2500  # 50 Hz * 50 s

    # Stream B: Spectrogram Parameters
    # Computed from original 200Hz signal
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 256  # Controls time resolution of spectrogram
    FMIN = 0.5  # Hz
    FMAX = 100.0  # Hz (Nyquist is 100Hz)

    # Spectrogram resizing for EfficientNet (Frequency, Time)
    # Note: Input to model will be (N_CHANNELS, SPEC_HEIGHT, SPEC_WIDTH)
    SPEC_HEIGHT = 64  # Matches N_MELS
    SPEC_WIDTH = 256  # Resized time dimension

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    PATIENCE = 3  # Early stopping patience
    NUM_WORKERS = 4  # For DataLoader

    # Debugging / Development
    DEBUG = False  # Set to True to use a small subset of data
    DEBUG_SAMPLE_SIZE = 1000


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
