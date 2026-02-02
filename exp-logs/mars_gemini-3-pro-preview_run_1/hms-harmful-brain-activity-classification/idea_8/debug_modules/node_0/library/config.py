import os
import torch
import numpy as np
import random


class Config:
    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Raw Data
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory for Idea 8
    WORKING_DIR = "./working/idea_8"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Configuration
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

    # EEG Constants
    EEG_SR = 200  # Original sampling rate (Hz)
    TARGET_SR = 100  # Downsampled rate for model (Hz)
    EEG_DURATION = 50  # Duration of the crop (seconds)
    EEG_SEQ_LEN = 5000  # 50 sec * 100 Hz
    EEG_CHANNELS = 20  # 19 EEG leads + 1 EKG

    # Spectrogram Constants
    SPEC_DURATION = 600  # 10 minutes (seconds)
    SPEC_SIZE = (512, 512)  # Input size (Time, Freq) for the 2D backbone

    # =========================================================================
    # Model Architecture (Chronologically-Embedded Dual-Stream Transformer)
    # =========================================================================
    # Stream A: Raw EEG (Multi-Scale 1D CNN)
    EEG_KERNELS = [3, 5, 7, 9]  # Kernel sizes for parallel branches
    EEG_MODEL_DIM = 256  # Feature dimension after CNN

    # Stream B: Spectrogram (EfficientNet)
    SPEC_BACKBONE = "tf_efficientnet_b0_ns"
    SPEC_PRETRAINED = True
    SPEC_EMBED_DIM = 256  # Projection dimension

    # Fusion: Transformer Decoder
    TRANSFORMER_LAYERS = 2
    TRANSFORMER_HEADS = 4
    TRANSFORMER_DIM = 256
    TRANSFORMER_FF_DIM = 1024
    DROPOUT = 0.1
    DROP_PATH_RATE = 0.1  # Stochastic depth regularization

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32  # Fits comfortably on A100-40GB
    NUM_WORKERS = 8  # 12 vCPUs available
    EPOCHS = 12  # Max epochs (with early stopping)

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01
    MAX_GRAD_NORM = 1.0
    PATIENCE = 4  # Early stopping patience

    # Scheduler (OneCycleLR)
    PCT_START = 0.1  # Warmup percentage
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # =========================================================================
    # Development & Debugging
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 1000  # Number of samples for debug run

    @classmethod
    def setup(cls):
        """
        Initializes the environment: creates necessary directories and sets
        random seeds for reproducibility.
        """
        # Create directories
        for d in [
            cls.WORKING_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.SUBMISSION_DIR,
        ]:
            os.makedirs(d, exist_ok=True)

        # Set Random Seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
