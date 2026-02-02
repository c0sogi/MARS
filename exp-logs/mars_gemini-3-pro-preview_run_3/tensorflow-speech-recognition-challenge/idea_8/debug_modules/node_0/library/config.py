import os
import torch


class Config:
    # ==========================================
    # System & Paths
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Input Directories
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories (Idea 8 Specific)
    WORKING_DIR = "./working/idea_8"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # File Paths
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio Processing (Multi-Resolution)
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    N_SAMPLES = int(SAMPLE_RATE * DURATION)

    # STFT Parameters
    N_MELS = 64
    HOP_LENGTH = 160  # 10ms hop
    F_MIN = 20
    F_MAX = 8000

    # Multi-Resolution Windows (Short, Medium, Long)
    # 20ms, 40ms, 60ms -> Maps to RGB channels
    WINDOW_SIZES_SEC = [0.02, 0.04, 0.06]
    WINDOW_SIZES = [int(ws * SAMPLE_RATE) for ws in WINDOW_SIZES_SEC]

    # ==========================================
    # Dataset & Labels
    # ==========================================
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
    LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
    IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}

    # Debugging / Sampling control
    MAX_TRAIN_SAMPLES = None  # Use None for full dataset
    MAX_VAL_SAMPLES = None

    # ==========================================
    # Model Architecture (SK-ResNet-Conformer)
    # ==========================================
    BACKBONE = "skresnet34"
    PRETRAINED = True
    IN_CHANNELS = 3  # RGB mapping of multi-res spectrograms

    # Conformer Neck
    CONFORMER_DIM = 512  # Matches ResNet34 expansion
    CONFORMER_HEADS = 8
    CONFORMER_LAYERS = 2
    CONFORMER_DROPOUT = 0.1

    # Attention Pooling Head
    POOLING_HEADS = 4

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 64
    NUM_EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler
    T_MAX = NUM_EPOCHS  # For CosineAnnealing
    MIN_LR = 1e-6

    # ==========================================
    # Augmentation (SpecAugment)
    # ==========================================
    # Time mask < 20% of duration
    # Time steps = 1 + N_SAMPLES // HOP_LENGTH = 101
    # 20% of 101 is ~20.
    TIME_MASK_PARAM = 20
    FREQ_MASK_PARAM = 10

    @staticmethod
    def setup():
        """Creates necessary directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
