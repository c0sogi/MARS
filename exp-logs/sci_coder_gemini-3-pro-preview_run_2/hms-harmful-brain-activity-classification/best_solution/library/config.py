import os
import torch
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # General Configuration
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    NUM_CLASSES = 6
    CLASS_NAMES = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    # -------------------------------------------------------------------------
    # Compute & Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use available CPUs for data loading
    NUM_WORKERS = os.cpu_count()

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for checkpoints, cache, and submissions
    OUTPUT_DIR = "./working/idea_2"
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_FILE = "./submission/submission.csv"

    # Raw Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # -------------------------------------------------------------------------
    # Data Processing: EEG (Stream 1)
    # -------------------------------------------------------------------------
    EEG_SR = 200  # Sampling rate in Hz
    EEG_DURATION = 50  # Duration in seconds
    EEG_SEQ_LEN = EEG_SR * EEG_DURATION  # 10,000 samples

    # Longitudinal Bipolar Montage (Double Banana) - 16 Channels
    # Standard 10-20 System Channel Names in the dataset
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
        "EKG",
    ]

    # Pairs for calculating differential signals
    MONTAGE_PAIRS = [
        ("Fp1", "F7"),
        ("F7", "T3"),
        ("T3", "T5"),
        ("T5", "O1"),  # Left Temporal
        ("Fp2", "F8"),
        ("F8", "T4"),
        ("T4", "T6"),
        ("T6", "O2"),  # Right Temporal
        ("Fp1", "F3"),
        ("F3", "C3"),
        ("C3", "P3"),
        ("P3", "O1"),  # Left Parasagittal
        ("Fp2", "F4"),
        ("F4", "C4"),
        ("C4", "P4"),
        ("P4", "O2"),  # Right Parasagittal
    ]

    # Mel Spectrogram Parameters for EEG
    N_FFT = 1024
    HOP_LENGTH = 512 // 2  # Overlap to get decent width
    N_MELS = 128
    FMIN = 0.5
    FMAX = 40.0  # Focus on medically relevant frequencies (Delta to Gamma)

    # -------------------------------------------------------------------------
    # Data Processing: Kaggle Spectrograms (Stream 2)
    # -------------------------------------------------------------------------
    SPEC_DURATION = 600  # 10 minutes
    # The raw parquet files have varying shapes, will be resized

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE = "efficientnet_b1"
    PRETRAINED = True
    IMG_SIZE = (512, 512)  # Input size for the EfficientNet
    IN_CHANNELS = 3  # We will replicate mono/diff channels to RGB or use 1 if modified

    # Regularization
    DROP_RATE = 0.2  # Head dropout
    DROP_PATH_RATE = 0.2  # Stochastic depth

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 6
    LR = 1e-3
    WEIGHT_DECAY = 1e-2

    # Augmentation
    MIXUP_ALPHA = 0.4
    MASK_MAX_SIZE = 32  # For CoarseDropout
    MASK_NUM_HOLES = 8

    # Optimization
    PATIENCE = 3  # Early stopping patience
    USE_AMP = True  # Automatic Mixed Precision

    # -------------------------------------------------------------------------
    # Setup Logic
    # -------------------------------------------------------------------------
    @classmethod
    def setup(cls):
        """Ensures output directories exist."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_FILE), exist_ok=True)


# Execute setup immediately upon import to guarantee paths exist
Config.setup()
