import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration class for the Right Whale Detection pipeline.
    Contains all file paths, hyperparameters, and global settings.
    """

    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    PROJECT_NAME = "RightWhaleDetection_Idea2"
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging/testing
    MAX_DEBUG_SAMPLES = 1000  # Number of samples to use if DEBUG is True

    # Compute
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # --------------------------------------------------------------------------
    # Paths
    # --------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching processed data and saving models
    WORKING_DIR = "./working/idea_2"

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # --------------------------------------------------------------------------
    # Audio Processing Parameters
    # --------------------------------------------------------------------------
    # Based on data analysis: SR=2000Hz, Max Duration=2.0s
    SAMPLE_RATE = 2000
    DURATION = 2.0  # Seconds

    # Spectrogram Parameters
    # Target shape approx (128, 235) to fit EfficientNet
    N_MELS = 128
    N_FFT = 256  # Window size: 256/2000 = 0.128s
    HOP_LENGTH = 16  # Hop size: 16/2000 = 0.008s -> ~250 frames for 2s audio
    FMIN = 20  # Min frequency
    FMAX = 1000  # Nyquist limit for 2kHz SR

    # --------------------------------------------------------------------------
    # Model Architecture
    # --------------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b0"
    IN_CHANNELS = 1
    NUM_CLASSES = 1
    USE_GEM_POOLING = True  # Generalized Mean Pooling for transient event detection

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 128
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5

    # Augmentation
    MIXUP_ALPHA = 0.2
    SPECAUG_TIME_MASK_PARAM = 20
    SPECAUG_FREQ_MASK_PARAM = 15

    @staticmethod
    def initialize():
        """
        Initializes the environment by setting random seeds and creating necessary directories.
        """
        # Set fixed random seeds for reproducibility
        seed = Config.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Create working and submission directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Print device info
        print(f"Configuration initialized. Device: {Config.DEVICE}")
        print(f"Working Directory: {Config.WORKING_DIR}")


# Automatically initialize environment when config is imported
Config.initialize()
