import os
import torch


class Config:
    # ==========================================
    # General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False
    EXPERIMENT_NAME = "idea_10"

    # ==========================================
    # Paths
    # ==========================================
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test2")

    # Metadata
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sampleSubmission.csv")

    # Output & Caching
    WORKING_DIR = "./working"
    OUTPUT_DIR = os.path.join(WORKING_DIR, EXPERIMENT_NAME)
    CACHE_DIR = OUTPUT_DIR  # Directory for cached processed data (npz/parquet)
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio Parameters
    # ==========================================
    SR = 2000  # Sampling Rate (Hz)
    DURATION = 2.0  # Target duration in seconds
    N_MELS = 320  # High resolution Mel bands
    N_FFT = 1024  # FFT window size
    WIN_LENGTH = 256  # Window length (~128ms)
    HOP_LENGTH = 16  # Hop length (~8ms) for high temporal resolution
    FMIN = 10  # Min frequency
    FMAX = 1000  # Max frequency (Nyquist is 1000Hz)

    # Data Preprocessing Flags
    FREQ_NORM = True  # Per-instance Frequency-Wise Standardization

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "convnext_small"
    PRETRAINED = True
    IN_CHANNELS = 1
    NUM_CLASSES = 1

    # Architectural Innovations
    USE_COORD_ATTN = True  # Coordinate Attention blocks
    USE_GEM_POOL = True  # Generalized Mean Pooling
    DROP_PATH_RATE = 0.2  # Stochastic Depth rate

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 20
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimizer & Scheduler
    LR = 1e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 3

    # Loss & Regularization
    USE_CLASS_WEIGHTS = True  # Inverse class frequency weighting
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4

    USE_SPECAUG = True
    SPECAUG_TIME_MASK = 30
    SPECAUG_FREQ_MASK = 40

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
