import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the Spectrogram-Guided Attentive Dual-Stream Network.
    Centralizes all hyperparameters, file paths, and system settings.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Output directory for checkpoints and cached data specific to this idea
    OUTPUT_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Raw Data Directories
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata Files (Generated in ./metadata)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Final Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Dataset & Preprocessing
    # =========================================================================
    SEED = 42
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

    # EEG Signal Processing
    EEG_SR = 100  # Target sampling rate (downsampled from 200Hz)
    EEG_DURATION = 50  # Duration in seconds
    EEG_SEQ_LEN = 5000  # 50 sec * 100 Hz
    EEG_CHANNELS = 20  # Number of EEG electrodes (19 EEG + 1 EKG)

    # Spectrogram Processing
    SPEC_SIZE = (512, 512)  # Input size for 2D CNN (Height, Width)

    # =========================================================================
    # Model Architecture
    # =========================================================================
    # Stream A: Multi-Scale 1D CNN (EEG)
    EEG_KERNEL_SIZES = [3, 5, 7, 9]
    EEG_BASE_FILTERS = 64

    # Stream B: 2D CNN (Spectrogram)
    SPEC_BACKBONE = "tf_efficientnet_b0_ns"
    PRETRAINED = True

    # Fusion Module (Cross-Attention)
    ATTENTION_DIM = 256
    DROPOUT_RATE = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    DEBUG = False  # Set True to run on a small subset for testing
    DEBUG_SUBSET_SIZE = 1000

    BATCH_SIZE = 32  # Optimized for A100 GPU (40GB VRAM)
    EPOCHS = 10  # Total training epochs

    # Optimizer (AdamW)
    LR = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 10.0

    # Scheduler (OneCycleLR)
    PCT_START = 0.1
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # Early Stopping
    PATIENCE = 3

    # System Settings
    NUM_WORKERS = 8  # Number of dataloader workers
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MIXED_PRECISION = True  # Enable FP16 training

    @classmethod
    def setup(cls):
        """
        Sets up the environment: creates directories and sets random seeds.
        """
        # Create output directories
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set random seeds for reproducibility
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Suppress warnings
        import warnings

        warnings.filterwarnings("ignore")
