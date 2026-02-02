import os
import torch


class Config:
    # ==========================================
    # Paths
    # ==========================================
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")
    BACKGROUND_NOISE_DIR = os.path.join(TRAIN_AUDIO_DIR, "_background_noise_")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working/idea_2"
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio Parameters
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # Seconds
    N_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Spectrogram parameters
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 160  # 10ms hop
    F_MIN = 20
    F_MAX = 8000  # Nyquist frequency

    # ==========================================
    # Labels
    # ==========================================
    # Core commands to detect
    COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

    # Full label set including special classes
    # Order is important for consistency
    LABELS = COMMANDS + ["silence", "unknown"]
    NUM_CLASSES = len(LABELS)

    # Mappings
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5  # Early stopping patience

    # ==========================================
    # Augmentation Parameters
    # ==========================================
    # MixUp
    MIXUP_ALPHA = 0.2

    # SpecAugment
    FREQ_MASK_PARAM = 20
    TIME_MASK_PARAM = 20  # < 20% of time steps

    # ==========================================
    # Compute
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Matches available vCPUs
