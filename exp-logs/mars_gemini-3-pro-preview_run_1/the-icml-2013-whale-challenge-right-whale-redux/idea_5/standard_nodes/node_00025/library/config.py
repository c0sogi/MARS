import os
import torch


class Config:
    # ==========================================
    # Project Metadata
    # ==========================================
    PROJECT_NAME = "right_whale_detection"
    IDEA_NAME = "idea_5"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing

    # ==========================================
    # Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching intermediate data (spectrograms, etc.)
    WORKING_DIR = os.path.join("./working", IDEA_NAME)

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # ==========================================
    # Audio Parameters
    # ==========================================
    SAMPLE_RATE = 2000
    DURATION = 2.0  # Seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)  # 4000 samples

    # Spectrogram Generation
    N_FFT = 512  # Gives 257 frequency bins (0-1000Hz)
    # Window size: 25ms * 2000Hz = 50 samples
    WIN_LENGTH = 50
    # Hop length: 10ms * 2000Hz = 20 samples
    HOP_LENGTH = 20
    N_MELS = 128
    F_MIN = 20  # High-pass filter
    F_MAX = 1000  # Nyquist frequency

    # ==========================================
    # Model Parameters
    # ==========================================
    BACKBONE_NAME = "efficientnet_b0"
    PRETRAINED = True

    # RNN / Temporal Modeling
    RNN_HIDDEN_SIZE = 128
    RNN_LAYERS = 1
    BIDIRECTIONAL = True

    # Attention Pooling
    ATTENTION_DIM = 128

    DROPOUT = 0.2
    NUM_CLASSES = 1

    # ==========================================
    # Training Parameters
    # ==========================================
    BATCH_SIZE = 64
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # Scheduler
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Data Augmentation
    MIXUP_ALPHA = 0.4

    # SpecAugment
    # Time mask: Max 200ms.
    # 200ms / 10ms hop = 20 frames.
    TIME_MASK_PARAM = 20
    FREQ_MASK_PARAM = 16  # Approx 1/8th of n_mels

    # ==========================================
    # Compute / Environment
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Ensure necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Print the current configuration."""
        print(f"Configuration for {cls.IDEA_NAME}:")
        print(f"  Device: {cls.DEVICE}")
        print(f"  Sample Rate: {cls.SAMPLE_RATE}, Duration: {cls.DURATION}s")
        print(
            f"  Spectrogram: {cls.N_MELS} Mels, {cls.WIN_LENGTH} Win, {cls.HOP_LENGTH} Hop"
        )
        print(f"  Model: {cls.BACKBONE_NAME} + BiGRU({cls.RNN_HIDDEN_SIZE})")
        print(
            f"  Training: BS={cls.BATCH_SIZE}, LR={cls.LEARNING_RATE}, Mixup={cls.MIXUP_ALPHA}"
        )
