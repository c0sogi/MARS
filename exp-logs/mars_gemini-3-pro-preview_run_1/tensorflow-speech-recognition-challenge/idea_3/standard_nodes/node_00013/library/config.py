import os
import torch


class Config:
    """
    Centralized configuration for the Speech Commands classification task.
    Includes paths, audio processing parameters, and training hyperparameters.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    PROJECT_NAME = "speech_commands_efficientnet"
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and models (Idea 3 specific)
    WORKING_DIR = "./working/idea_3"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # -------------------------------------------------------------------------
    # Audio Signal Processing (Spectrogram Generation)
    # -------------------------------------------------------------------------
    SAMPLE_RATE = 16000
    DURATION = 1.0  # Duration of clips in seconds
    N_MELS = 128  # Number of Mel bands
    N_FFT = 1024  # FFT window size (~64ms)
    HOP_LENGTH = 160  # Hop length (~10ms)
    F_MIN = 0.0
    F_MAX = None  # Defaults to SAMPLE_RATE // 2

    # -------------------------------------------------------------------------
    # Labels & Mappings
    # -------------------------------------------------------------------------
    # The 10 target commands to be predicted
    TARGET_LABELS = [
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ]

    # Full label set: Targets + Silence + Unknown
    # Order is important for consistency across runs
    LABELS = TARGET_LABELS + ["silence", "unknown"]
    NUM_CLASSES = len(LABELS)

    # Mappings for conversion
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    EPOCHS = 40  # Extended training as per strategy
    BATCH_SIZE = 128  # Optimized for A100 GPU
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Standard for AdamW

    # Augmentation
    MIXUP_ALPHA = 0.4  # Strong mixup regularization

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # -------------------------------------------------------------------------
    # Hardware & Execution
    # -------------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of dataloader workers (12 vCPUs available)

    # -------------------------------------------------------------------------
    # Debugging & Development
    # -------------------------------------------------------------------------
    # Set DEBUG to True to run on a small subset of data for quick testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Initialize the environment by creating necessary directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic behavior for reproducibility
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
