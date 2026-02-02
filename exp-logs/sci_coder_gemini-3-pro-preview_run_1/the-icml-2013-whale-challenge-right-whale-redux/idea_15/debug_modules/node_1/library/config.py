import os


class AudioConfig:
    """
    Configuration for Audio Processing.
    Prioritizes high frequency resolution (N_FFT=1024) while maintaining
    temporal resolution via high overlap (HOP_LENGTH=40).
    """

    SAMPLE_RATE = 2000
    DURATION = 2.0  # Seconds
    N_FFT = 1024  # Large window for ~2Hz frequency resolution
    HOP_LENGTH = 40  # Small hop (20ms) to generate ~100 time frames for RNN
    N_MELS = 128  # Number of Mel bands
    F_MIN = 10  # Min frequency (Hz)
    F_MAX = 1000  # Max frequency (Hz) - Nyquist
    TOP_DB = 80  # For Log-Mel conversion


class TrainConfig:
    """
    Configuration for Training Loop and Hyperparameters.
    """

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Ensemble Strategy
    SEEDS = [42, 101, 202, 303, 404]

    # Imbalance Handling
    POS_WEIGHT = 9.0  # Weight for positive class in BCE Loss

    # Mixup
    MIXUP_ALPHA = 0.4

    # Early Stopping
    PATIENCE = 7


class AugmentConfig:
    """
    Configuration for Data Augmentation.
    """

    # SpecAugment Constraints
    # Time mask width < 200ms.
    # With HOP_LENGTH=40 (20ms), 200ms is 10 frames.
    TIME_MASK_PARAM = 10
    FREQ_MASK_PARAM = 20


class ModelConfig:
    """
    Configuration for the Neural Network Architecture.
    """

    NAME = "SpecFPN_CRNN"
    BACKBONE = "resnet18"
    PRETRAINED = True
    IN_CHANNELS = 1

    # RNN Head
    RNN_HIDDEN_SIZE = 128
    RNN_LAYERS = 2
    RNN_DROPOUT = 0.3

    # SpecFPN
    FPN_OUT_CHANNELS = 256


class PathConfig:
    """
    Configuration for File Paths and Directories.
    """

    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Audio Directories
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test2")

    # Output Files
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    @classmethod
    def setup_directories(cls):
        """Ensure necessary writable directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
PathConfig.setup_directories()
