import os


class Config:
    """
    Configuration class for the Right Whale Detection task.
    Centralizes all file paths, audio processing parameters, and training hyperparameters.
    """

    # -------------------------------------------------------------------------
    # Directory and File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    TRAIN_DIR = os.path.join(INPUT_DIR, "train2")
    TEST_DIR = os.path.join(INPUT_DIR, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working"
    # Directory for caching processed data (e.g., spectrograms)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Audio Processing Parameters
    # -------------------------------------------------------------------------
    # Sample rate is 2000Hz as per dataset analysis
    SAMPLE_RATE = 2000

    # Duration of clips is 2.0 seconds
    DURATION = 2.0

    # Spectrogram parameters
    # Window size ~128ms (256 samples at 2000Hz)
    N_FFT = 256
    # 50% overlap
    HOP_LENGTH = 128
    # Number of Mel bands
    N_MELS = 64

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    EPOCHS = 20

    # Early stopping patience
    PATIENCE = 8

    # Augmentation
    AUGMENT = True
    FREQ_MASK_PARAM = 8
    TIME_MASK_PARAM = 4

    # Class imbalance handling
    # Positive class (whale calls) is the minority (~10%)
    # Weight = Majority Count / Minority Count ~= 9.0
    POS_WEIGHT = 9.0

    # -------------------------------------------------------------------------
    # Compute & Debugging
    # -------------------------------------------------------------------------
    NUM_WORKERS = 4

    # Debugging flags to control dataset size
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500

    @staticmethod
    def setup():
        """
        Creates necessary output directories if they do not exist.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
