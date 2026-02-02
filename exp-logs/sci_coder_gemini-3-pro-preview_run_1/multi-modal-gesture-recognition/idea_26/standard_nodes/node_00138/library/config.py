import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the SAMP-Net pipeline.
    Stores all hyperparameters, file paths, and constants.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific cache directory for deterministic data processing
    CACHE_DIR = "./working/idea_26"

    SUBMISSION_DIR = "./submission"
    CHECKPOINT_DIR = "./checkpoints"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Audio
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_FPS = 20  # Video FPS
    # Physics-Based Hop Length: SampleRate / VideoFPS = 16000 / 20 = 800
    AUDIO_HOP_LENGTH = 800
    AUDIO_N_FFT = 2048
    AUDIO_N_MFCC = 13  # Compact MFCCs

    # Skeleton
    NUM_JOINTS = 20
    CHANNELS_PER_JOINT = 3  # (x, y, z)
    SKELETON_INPUT_DIM = NUM_JOINTS * CHANNELS_PER_JOINT  # 60 features

    # -------------------------------------------------------------------------
    # Model Architecture Parameters
    # -------------------------------------------------------------------------
    AUDIO_INPUT_DIM = AUDIO_N_MFCC
    HIDDEN_DIM = 256
    KERNEL_SIZE = 7
    DROPOUT = 0.1

    # Classes: 20 gestures + 1 background (index 0)
    # Mapping: Background=0, Gestures=1..20
    NUM_CLASSES = 21

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.05
    EPOCHS = 50

    # Loss Weights
    AUX_LOSS_WEIGHT = 0.2
    BACKGROUND_CLASS_WEIGHT = 0.5
    LABEL_SMOOTHING = 0.1

    # -------------------------------------------------------------------------
    # Inference & Post-processing
    # -------------------------------------------------------------------------
    MEDIAN_FILTER_KERNEL = 5
    MIN_SEGMENT_LENGTH = 5

    # -------------------------------------------------------------------------
    # Hardware & Execution
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def setup():
        """
        Initializes the environment:
        1. Creates necessary directories (Cache, Submission, Checkpoints).
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

        # Set seeds
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
