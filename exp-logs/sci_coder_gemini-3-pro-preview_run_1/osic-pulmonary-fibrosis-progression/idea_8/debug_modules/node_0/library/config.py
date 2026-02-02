import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration module for the Residual Cross-Attention Dual-Axis Network.
    Handles file paths, hyperparameters, and global settings.
    """

    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    PROJECT_NAME = "lung_decline_prediction"
    EXPERIMENT_NAME = "idea_8"  # Specific identifier for caching
    SEED = 42
    DEBUG = False  # Set to True to run on a subset for quick debugging

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Metadata Paths (Pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    CACHE_DIR = os.path.join(WORKING_DIR, EXPERIMENT_NAME)
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing / Image Config
    # -------------------------------------------------------------------------
    IMG_SIZE = 224
    SLAB_COUNT = 3
    SLAB_OVERLAP = 0.15
    IN_CHANNELS = 3  # RGB channels from Tri-Slab MIPs

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    BACKBONE_NAME = "efficientnet_b0"
    PRETRAINED = True

    # Tabular Input Dimension Calculation:
    # Age (1) + Sex (OneHot=2) + SmokingStatus (OneHot=3) + Baseline_Percent (1) = 7
    TABULAR_INPUT_DIM = 7
    TABULAR_HIDDEN_DIM = 128
    ATTENTION_DIM = 64

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 4

    # SWA (Stochastic Weight Averaging) Settings
    USE_SWA = True
    SWA_START_EPOCH = 20
    SWA_LR = 1e-4

    # Early Stopping
    PATIENCE = 10

    # -------------------------------------------------------------------------
    # Metric / Loss Constants
    # -------------------------------------------------------------------------
    MAX_ERROR = 1000.0
    SIGMA_CLIP = 70.0

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup_directories():
        """Creates necessary output directories if they don't exist."""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def seed_everything(seed=42):
        """Sets random seeds for reproducibility."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
