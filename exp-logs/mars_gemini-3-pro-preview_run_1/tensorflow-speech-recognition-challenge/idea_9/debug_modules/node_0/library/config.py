import os
import torch
import random
import numpy as np


class Config:
    # --------------------------------------------------------------------------
    # Paths & Directories
    # --------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Cache Directory for this specific idea/experiment
    # Used to store processed spectrograms or intermediate features
    CACHE_DIR = "./working/idea_9"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Audio Parameters (Dual-Channel Architecture)
    # --------------------------------------------------------------------------
    SR = 16000
    DURATION = 1.0  # Seconds
    AUDIO_LEN_SAMPLES = int(SR * DURATION)

    # Mel Spectrogram Shared Params
    N_MELS = 128
    HOP_LENGTH = 160  # 10ms at 16kHz

    # Channel 1: High-Frequency Resolution (Formants)
    # Window size ~64ms
    N_FFT_FREQ = 1024

    # Channel 2: High-Temporal Resolution (Transients)
    # Window size ~25ms
    N_FFT_TIME = 400

    # Noise Injection Parameters
    NOISE_SNR_MIN = 10
    NOISE_SNR_MAX = 30
    NOISE_PROB = 0.5

    # --------------------------------------------------------------------------
    # Labels & Classes
    # --------------------------------------------------------------------------
    # The 10 specific commands to identify
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

    # Special labels
    SILENCE_LABEL = "silence"
    UNKNOWN_LABEL = "unknown"  # For submission output only

    # Auxiliary commands (sources of 'unknown' class)
    # Comprehensive list from Speech Commands V1/V2 to ensure coverage of the "unknown" source
    AUX_LABELS = [
        "bed",
        "bird",
        "cat",
        "dog",
        "eight",
        "five",
        "four",
        "happy",
        "house",
        "marvin",
        "nine",
        "one",
        "seven",
        "sheila",
        "six",
        "three",
        "tree",
        "two",
        "zero",
        "wow",
        "backward",
        "follow",
        "forward",
        "learn",
        "visual",
    ]

    # Fine-grained classes for training: Targets + Aux + Silence
    # We sort to ensure deterministic ID mapping across runs
    ALL_LABELS = sorted(TARGET_LABELS + AUX_LABELS) + [SILENCE_LABEL]
    NUM_CLASSES = len(ALL_LABELS)

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 128  # Optimized for A100 GPU (40GB VRAM)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MIXUP_ALPHA = 1.0

    # Data Subset for debugging (set to None for full training)
    DEBUG_SUBSET_SIZE = None

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------------------------
    # Utilities
    # --------------------------------------------------------------------------
    @classmethod
    def get_label2id(cls):
        """Returns dictionary mapping label string to integer ID."""
        return {label: i for i, label in enumerate(cls.ALL_LABELS)}

    @classmethod
    def get_id2label(cls):
        """Returns dictionary mapping integer ID to label string."""
        return {i: label for i, label in enumerate(cls.ALL_LABELS)}

    @classmethod
    def map_prediction_to_submission(cls, label):
        """
        Maps the predicted fine-grained label to the competition submission format.
        Rules:
        - If label is in TARGET_LABELS -> keep it.
        - If label is SILENCE_LABEL -> keep it.
        - Otherwise (Aux labels) -> 'unknown'.
        """
        if label in cls.TARGET_LABELS:
            return label
        elif label == cls.SILENCE_LABEL:
            return cls.SILENCE_LABEL
        else:
            return cls.UNKNOWN_LABEL


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When using CuDNN, these settings ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
