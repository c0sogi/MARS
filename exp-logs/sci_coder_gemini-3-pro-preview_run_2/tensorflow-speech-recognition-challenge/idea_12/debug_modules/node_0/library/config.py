import os
import random
import numpy as np
import torch


class Config:
    """
    Configuration module for Speech Command Recognition (Idea 12).
    Implements settings for EfficientNetV2-B0 with High-Fidelity Signal Pipeline.
    """

    # ==========================================
    # 1. General Configuration
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to use a small subset for debugging
    DEBUG_SUBSET_SIZE = 500
    NUM_WORKERS = 6  # Optimized for 12 vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 2. File Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Input Data Directories
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")
    BACKGROUND_NOISE_DIR = os.path.join(TRAIN_AUDIO_DIR, "_background_noise_")

    # Metadata CSV Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Audio Processing (High-Fidelity Pipeline)
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # Seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Spectrogram Generation Parameters
    # Idea 12: Spectral Oversampling (1024 FFT) with 25ms window and 10ms hop
    N_FFT = 1024  # Larger FFT for interpolation
    WIN_LENGTH = 400  # 25ms
    HOP_LENGTH = 160  # 10ms
    N_MELS = 128  # High frequency resolution
    F_MIN = 20
    F_MAX = 7800  # Near Nyquist (8000Hz)

    # ==========================================
    # 4. Labels
    # ==========================================
    # Target classes as per competition spec
    LABELS = [
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
        "silence",
        "unknown",
    ]
    NUM_CLASSES = len(LABELS)
    LABEL2IDX = {label: idx for idx, label in enumerate(LABELS)}
    IDX2LABEL = {idx: label for idx, label in enumerate(LABELS)}

    # ==========================================
    # 5. Model Architecture
    # ==========================================
    MODEL_NAME = "efficientnet_v2_b0"
    IN_CHANNELS = 1  # Log-Mel Spectrogram is 1-channel
    DROPOUT = 0.2

    # ==========================================
    # 6. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    LABEL_SMOOTHING = 0.1

    # Exponential Moving Average (EMA)
    USE_EMA = True
    EMA_DECAY = 0.999

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # ==========================================
    # 7. Augmentation Strategy
    # ==========================================
    # Waveform Noise Injection
    NOISE_INJECTION_PROB = 0.5
    NOISE_MIN_SNR_DB = 5.0
    NOISE_MAX_SNR_DB = 30.0

    # SpecAugment (Time/Freq Masking)
    TIME_MASK_PARAM = 20  # Max time steps to mask
    FREQ_MASK_PARAM = 20  # Max freq bins to mask

    @classmethod
    def initialize(cls):
        """
        Sets up the environment: creates necessary directories and sets random seeds.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        cls.set_seed(cls.SEED)

    @staticmethod
    def set_seed(seed):
        """
        Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize configuration environment on import
Config.initialize()
