import os
import torch


class Config:
    """
    Configuration for the Decoupled Dual-Stream Residual Network (DDSR-Net).
    """

    # ==========================
    # General Settings
    # ==========================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging
    DEBUG = False  # Set to True to use a smaller subset of data
    DEBUG_SIZE = 50  # Number of samples to use in debug mode

    # ==========================
    # File Paths
    # ==========================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Idea-specific working directory
    WORKING_DIR = "./working/idea_14"

    # Subdirectories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # ==========================
    # Data Hyperparameters
    # ==========================
    IMG_SIZE = 384
    NUM_SLICES = 3  # Anchor + 2 boundaries

    # Normalization Stats (Derived from EDA)
    # Target (FVC)
    TARGET_MEAN = 2654.6528
    TARGET_STD = 801.7017

    # Features
    AGE_MEAN = 67.5825
    AGE_STD = 6.6259

    # Note: Baseline FVC stats are approximately the same as Target FVC for normalization purposes
    BASE_FVC_MEAN = 2654.6528
    BASE_FVC_STD = 801.7017

    # ==========================
    # Model Hyperparameters
    # ==========================
    BACKBONE_NAME = "tf_efficientnetv2_s"
    BACKBONE_PRETRAINED = True

    # Projection dimensions
    IMG_EMBED_DIM = 128
    TABULAR_EMBED_DIM = 128
    HIDDEN_DIM = 128

    # ==========================
    # Training Hyperparameters
    # ==========================
    EPOCHS = 50
    BATCH_SIZE = 32

    # Differential Learning Rates
    LR_BACKBONE = 1e-4
    LR_HEADS = 1e-3

    # Optimizer & Scheduler
    WEIGHT_DECAY = 1e-2
    T_MAX = 50  # Cosine Annealing duration
    ETA_MIN = 1e-6  # Minimum LR

    # Loss / Metric
    SIGMA_MIN = 70.0
    MAX_ERROR = 1000.0

    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)
