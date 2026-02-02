import os
import torch


class Config:
    # =========================================================================
    # General Settings
    # =========================================================================
    PROJECT_NAME = "BirdSpeciesClassification_ResNet34_Attn"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset of data for testing

    # =========================================================================
    # Directory Paths
    # =========================================================================
    # Root input directory (Read-Only)
    INPUT_DIR = "./input"

    # Metadata directory (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for cache and intermediate files
    # Using 'idea_4' as per the prompt's context for the current iteration
    WORK_DIR = "./working/idea_4"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Audio Processing Parameters
    # =========================================================================
    SR = 16000  # Sampling rate (16kHz)
    DURATION = 10  # Duration in seconds
    N_MELS = 128  # Number of Mel bands
    N_FFT = 1024  # FFT window size
    HOP_LENGTH = 320  # Hop length (results in ~500 time frames for 10s)
    F_MIN = 0  # Min frequency
    F_MAX = 8000  # Max frequency (Nyquist)

    # Image dimensions for the model input (Freq, Time)
    # We keep the aspect ratio rectangular to preserve temporal resolution
    IMG_HEIGHT = N_MELS
    IMG_WIDTH = 501  # Approx 16000*10 / 320 + 1

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    MODEL_NAME = "resnet34"
    PRETRAINED = True
    NUM_CLASSES = 19
    IN_CHANNELS = 3  # Replicating mono spectrogram to 3 channels

    # Attention Pooling Specifics
    USE_ATTENTION_POOLING = True
    ATTENTION_HIDDEN_DIM = 128

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    NUM_FOLDS = 5  # Stratified K-Fold

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 10

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """
        Creates necessary directories for working, caching, and submissions.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
