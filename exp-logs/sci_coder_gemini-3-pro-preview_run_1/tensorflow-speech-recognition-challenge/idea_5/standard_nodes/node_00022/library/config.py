import os
import torch


class Config:
    # ==========================================
    # Paths and Directories
    # ==========================================
    PROJECT_ROOT = "."
    INPUT_ROOT = os.path.join(PROJECT_ROOT, "input")
    METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")

    # Working directory for Idea 5 (Dilated ConvNeXt-Tiny)
    WORKING_DIR = os.path.join(PROJECT_ROOT, "working", "idea_5")
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_ROOT, "sample_submission.csv")

    # Audio File Paths
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    AUDIO_LEN = int(SAMPLE_RATE * DURATION)

    # Spectrogram Parameters (Sweet Spot from Idea 1 & 3)
    N_MELS = 128
    N_FFT = 1024  # 64ms window
    HOP_LENGTH = 160  # 10ms hop
    F_MIN = 20
    F_MAX = SAMPLE_RATE // 2

    # ==========================================
    # Data & Label Configuration
    # ==========================================
    # The 10 specific commands we need to detect
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

    # Full label set including auxiliary classes
    # Order matters for prediction mapping
    LABELS = TARGET_LABELS + ["silence", "unknown"]
    NUM_CLASSES = len(LABELS)

    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "convnext_tiny"
    PRETRAINED = True
    IN_CHANNELS = 1  # Spectrogram is 1 channel

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32
    EPOCHS = 50

    # Optimizer
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Regularization
    MIXUP_ALPHA = 1.0  # High mixup for ConvNeXt

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoint paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
