import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Config:
    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data/models
    # Using idea_17 for fresh training run with extended epochs
    WORKING_DIR = "./working/idea_17"

    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Audio Processing Parameters
    # ==========================================
    SR = 16000  # Sample Rate: 16kHz
    DURATION = 1.0  # Duration: 1 second
    NUM_SAMPLES = int(SR * DURATION)

    # Spectrogram Generation (Spectral Oversampling strategy)
    N_FFT = 1024  # FFT window size (interpolates spectrum)
    WIN_LEN = 400  # Window length: 25ms (0.025 * 16000)
    HOP_LEN = 160  # Hop length: 10ms (0.010 * 16000)
    N_MELS = 128  # High-resolution Mel bins
    F_MIN = 20  # Min frequency
    F_MAX = SR // 2  # Nyquist frequency

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    MODEL_NAME = "tf_efficientnetv2_b0"
    NUM_CLASSES = 12
    IN_CHANNELS = 1  # 1-channel input (spectrogram)

    # Target Labels in order
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
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32  # Small batch size for high update frequency (Cite Lesson 00031)
    NUM_EPOCHS = 45  # Extended epochs to maximize convergence (Cite Lesson 00042)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Optimization
    LABEL_SMOOTHING = 0.1
    USE_EMA = True  # Exponential Moving Average for weights
    EMA_DECAY = 0.999

    # Augmentation
    NOISE_INJECTION_PROB = 0.5
    SPEC_AUG_TIME_MASK = 20
    SPEC_AUG_FREQ_MASK = 20

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0  # 0 workers because data will be GPU-resident

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist and sets up environment.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Enable TF32 for A100 if available for speedup
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
