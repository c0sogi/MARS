import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_DIR, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_DIR, "test", "audio")
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_5"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output
    SUBMISSION_PATH = "./submission/submission.csv"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # Audio Parameters
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    N_SAMPLES = int(SAMPLE_RATE * DURATION)

    # ==========================================
    # Feature Extraction (Multi-Resolution)
    # ==========================================
    # We use 3 different window sizes for the 3 image channels (RGB equivalent)
    # 20ms, 40ms, 60ms at 16kHz
    WINDOW_SIZES = [320, 640, 960]
    HOP_LENGTH = 160  # 10ms fixed hop for alignment
    N_MELS = 80
    F_MIN = 20
    F_MAX = 8000

    # ==========================================
    # Labels
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
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True
    # RNN / Head settings
    HIDDEN_SIZE = 128
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 128
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    PATIENCE = 5  # Early stopping patience

    # ==========================================
    # Augmentation (SpecAugment)
    # ==========================================
    FREQ_MASK_PARAM = 10
    # Time mask must be < 20% of duration.
    # With hop=160, 1sec = 100 frames. 20% is 20 frames.
    TIME_MASK_PARAM = 15

    # ==========================================
    # Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @classmethod
    def setup(cls):
        """Ensures working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)
