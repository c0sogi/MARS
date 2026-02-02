import os
import torch
import numpy as np
import random


SR = 200


class Config:
    """
    Centralized configuration for the Multi-Resolution Dual-Stream Network experiment.
    """

    # ==========================================
    # General Setup
    # ==========================================
    PROJECT_NAME = "Brain_Activity_Classification_Idea_10"
    SEED = 42
    NUM_WORKERS = 8  # Optimized for 12 vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"

    # Ensure working directory exists for caching/checkpoints
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Data Sources
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Caching
    CACHE_DIR = WORKING_DIR

    # ==========================================
    # Data & Signal Processing
    # ==========================================
    # EEG Signal Constants
    SR = SR  # Sampling Rate in Hz
    DURATION = 50  # Seconds
    TOTAL_SAMPLES = SR * DURATION  # 10,000 samples

    # EEG Channels: Standard 10-20 system (19 channels), excluding EKG
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
    N_EEG_CHANNELS = len(EEG_CHANNELS)

    # Stream A: Multi-Resolution STFT Parameters
    # Window sizes in milliseconds: 32ms (Fast), 250ms (Medium), 1000ms (Slow)
    STFT_WINDOW_MS = [32, 250, 1000]
    # Convert to samples: [6, 50, 200] at 200Hz
    STFT_WINDOW_SIZES = [int(ms * SR / 1000) for ms in STFT_WINDOW_MS]

    # Mel Spectrogram Settings
    N_MELS = 128
    FMIN = 0.5
    FMAX = 100.0  # Nyquist frequency

    # Target Dimensions for Stream A (Time-Frequency)
    # Shape: (Frequency Bins, Time Steps) -> (128, 500)
    IMG_SIZE_A = (128, 500)

    # Target Dimensions for Stream B (Kaggle Spectrograms)
    # Shape: (Height, Width) -> (256, 256)
    IMG_SIZE_B = (256, 256)

    # ==========================================
    # Model Architecture
    # ==========================================
    # Stream A Backbone (Detail)
    MODEL_BACKBONE_A = "tf_efficientnet_b2"
    # Input channels: 19 EEG channels * 3 Resolutions = 57
    IN_CHANNELS_A = N_EEG_CHANNELS * len(STFT_WINDOW_SIZES)

    # Stream B Backbone (Context)
    MODEL_BACKBONE_B = "tf_efficientnet_b0"
    # Input channels: 4 regions (LL, RL, LP, RP)
    IN_CHANNELS_B = 4

    # Output
    NUM_CLASSES = 6
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

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0
    PATIENCE = 3  # For Early Stopping

    # Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    USE_SPECAUG = True
    SPECAUG_MASK_SIZE = 10

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False
    DEBUG_SUBSET_SIZE = 1000  # Number of samples to use when DEBUG=True


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
