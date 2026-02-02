import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Project & Path Configuration
    # ==========================================
    PROJECT_NAME = "idea_8"
    ROOT_DIR = "."

    # Input Directories
    INPUT_DIR = os.path.join(ROOT_DIR, "input")
    METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
    ESSENTIAL_DATA_DIR = os.path.join(INPUT_DIR, "essential_data")
    SUPPLEMENTAL_DATA_DIR = os.path.join(INPUT_DIR, "supplemental_data")

    # Specific Data Files/Folders
    SPECTROGRAM_DIR = os.path.join(SUPPLEMENTAL_DATA_DIR, "spectrograms")
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Directories
    WORKING_DIR = os.path.join(ROOT_DIR, "working", PROJECT_NAME)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 19
    IN_CHANNELS = 3  # RGB (Spectrogram replicated 3 times)

    # Multi-View MIL Strategy
    IMAGE_SIZE = 224
    NUM_TILES = 3  # Number of temporal crops per recording (Start, Middle, End)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    NUM_FOLDS = 5
    EPOCHS = 50
    BATCH_SIZE = 16  # Conservative batch size for stability
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2
    PATIENCE = 15  # Early stopping patience

    # Scheduler (Cosine Annealing)
    SCHEDULER_T_MAX = EPOCHS
    SCHEDULER_ETA_MIN = 1e-6

    # Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    AUG_BRIGHTNESS = 0.2
    AUG_CONTRAST = 0.2

    # ==========================================
    # Compute & Environment
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False
    DEBUG_SAMPLES = 50  # Limit dataset size when DEBUG is True

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary output directories.
        2. Set fixed random seeds for reproducibility.
        """
        # Create directories
        for d in [
            cls.WORKING_DIR,
            cls.CACHE_DIR,
            cls.CHECKPOINT_DIR,
            cls.SUBMISSION_DIR,
        ]:
            os.makedirs(d, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"Config initialized for {cls.PROJECT_NAME}")
        print(f"Directories created at {cls.WORKING_DIR}")
        print(f"Device: {cls.DEVICE}")
