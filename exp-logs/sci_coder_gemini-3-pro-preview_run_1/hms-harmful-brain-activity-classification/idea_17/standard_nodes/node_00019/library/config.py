import os
import torch


class Config:
    """
    Configuration class for the 'Coordinate-Focused Dual-Stream Network' (Idea 17).
    Centralizes all file paths, hyperparameters, and data dimensions.
    """

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Source Data
    TRAIN_EEGS_DIR = os.path.join(INPUT_DIR, "train_eegs")
    TRAIN_SPECS_DIR = os.path.join(INPUT_DIR, "train_spectrograms")
    TEST_EEGS_DIR = os.path.join(INPUT_DIR, "test_eegs")
    TEST_SPECS_DIR = os.path.join(INPUT_DIR, "test_spectrograms")

    # Metadata
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output / Working Directory (Idea 17 specific)
    WORKING_DIR = "./working/idea_17"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Parameters
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

    # EEG Configuration
    # Downsampling to 100Hz for 50s window -> 5000 time steps
    EEG_RAW_SR = 200
    EEG_TARGET_SR = 100
    EEG_DURATION_S = 50
    EEG_SEQ_LEN = EEG_DURATION_S * EEG_TARGET_SR  # 5000
    EEG_CHANNELS = 20  # 19 EEG electrodes + 1 EKG

    # Spectrogram Configuration
    # 10 minute window, resized to 512x512
    # Channels: 4 regions (LL, RL, LP, RP) + 1 Coordinate Map = 5
    SPEC_HEIGHT = 512  # Frequency axis
    SPEC_WIDTH = 512  # Time axis
    SPEC_CHANNELS = 5

    # =========================================================================
    # Training Strategy (Global Random Subsampling)
    # =========================================================================
    # Train on a fixed subset of 25,000 samples per run to ensure convergence
    # within time limits while maintaining batch norm stability.
    TRAIN_SAMPLE_SIZE = 25000

    EPOCHS = 5
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_LR = 1e-3
    PCT_START = 0.3  # For OneCycleLR

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Stream A: SE-Inception 1D
    INCEPTION_DEPTH = 6
    INCEPTION_KERNELS = [3, 5, 7]
    INCEPTION_FILTERS = 32

    # Stream B: Coordinate-Focused EfficientNet
    BACKBONE_2D = "efficientnet_b0"
    PRETRAINED = True

    # Fusion
    ATTENTION_HEADS = 4
    DROPOUT = 0.5

    # =========================================================================
    # Hardware
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def make_dirs(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
