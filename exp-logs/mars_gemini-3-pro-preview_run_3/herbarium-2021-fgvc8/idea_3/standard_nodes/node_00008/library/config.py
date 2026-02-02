import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Plant Species Classification task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for debugging
    DEBUG_SAMPLE_SIZE = 5000  # Number of samples to use when DEBUG is True

    # ==========================================
    # Directories & Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"

    # Metadata files (pre-generated)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model_best.pth")
    SWA_MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model_swa.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "convnext_small.fb_in1k"  # timm backbone
    NUM_CLASSES = 64500
    EMBEDDING_DIM = 768  # Output dimension of ConvNeXt-Small
    DROPOUT = 0.0
    DROP_PATH_RATE = 0.1  # Stochastic depth rate

    # ==========================================
    # Data Processing
    # ==========================================
    IMG_SIZE = 224
    BATCH_SIZE = 128
    NUM_WORKERS = 12

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 12
    LR = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.1
    GRAD_CLIP = 1.0

    # ==========================================
    # SWA (Stochastic Weight Averaging)
    # ==========================================
    USE_SWA = True
    SWA_START_EPOCH = 8
    SWA_LR = 5e-5

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """
        Performs initial setup:
        1. Creates the working directory.
        2. Sets random seeds for reproducibility.
        """
        # Create working directory
        os.makedirs(cls.WORKING_DIR, exist_ok=True)

        # Create submission directory if it doesn't exist
        submission_dir = os.path.dirname(cls.SUBMISSION_PATH)
        if submission_dir:
            os.makedirs(submission_dir, exist_ok=True)

        # Set seeds
        cls.set_seed()

    @classmethod
    def set_seed(cls):
        """Sets fixed random seeds for reproducibility."""
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Execute setup on import to ensure environment is ready
Config.setup()
