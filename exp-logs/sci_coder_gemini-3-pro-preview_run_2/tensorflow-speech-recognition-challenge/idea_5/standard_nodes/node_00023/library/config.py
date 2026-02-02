import os
import torch


class Config:
    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    # Root directories
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Specific file paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Audio directories (constructed relative to input root)
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

    # Output paths
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Processing Hyperparameters
    # ==========================================
    SEED = 42
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    MAX_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Spectrogram Parameters (High Resolution)
    # Window size: 25ms -> 0.025 * 16000 = 400 samples
    # Hop length: 10ms -> 0.010 * 16000 = 160 samples
    N_FFT = 400
    HOP_LENGTH = 160
    N_MELS = 128

    # Image Adaptation for Swin Transformer
    IMG_SIZE = 224  # Resize spectrograms to 224x224

    # SpecAugment Parameters (Calibrated)
    FREQ_MASK_PARAM = 20  # < 20% of 128 bins
    TIME_MASK_PARAM = 20  # < 20% of time steps (approx 100 steps)

    # ==========================================
    # 3. Model & Training Hyperparameters
    # ==========================================
    # Labels
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

    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01  # Standard for AdamW
    LABEL_SMOOTHING = 0.1

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging / Development
    # Set to a small integer (e.g., 1000) to run on a subset of data for quick testing
    # Set to None to run on the full dataset
    DEBUG_SUBSET_SIZE = None

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 30)
        print("CONFIG")
        print("=" * 30)
        for key, value in cls.__dict__.items():
            if not key.startswith("__") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 30)
