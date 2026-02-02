import os
import torch


class Config:
    """
    Centralized configuration for the Speech Command Recognition task.
    """

    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    SUBMISSION_DIR = "./submission"

    # Cache directory for deterministic data processing
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    # Metadata CSV paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Audio source directories (read-only)
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

    # Output paths
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "scb_best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Audio Processing Parameters
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Mel-Spectrogram features
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 512

    # ==========================================
    # 3. Label Configuration
    # ==========================================
    # The 12 target classes for the competition
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

    # Mappings
    LABEL2IDX = {label: idx for idx, label in enumerate(LABELS)}
    IDX2LABEL = {idx: label for idx, label in enumerate(LABELS)}

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 5

    # ==========================================
    # 5. Compute Settings
    # ==========================================
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 6. Debugging / Development
    # ==========================================
    # If True, datasets will be limited to DEBUG_SUBSET_SIZE for rapid testing
    DEBUG = False
    DEBUG_SUBSET_SIZE = 1000

    @staticmethod
    def setup():
        """
        Creates necessary output directories if they don't exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
