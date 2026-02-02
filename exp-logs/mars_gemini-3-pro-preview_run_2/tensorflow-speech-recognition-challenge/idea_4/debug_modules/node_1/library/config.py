import os
import torch


class Config:
    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching features and saving models
    WORKING_DIR = "./working/idea_4"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Model checkpoint path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==========================================
    # 2. Audio Processing Parameters
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Spectrogram Parameters (High Temporal Resolution)
    # Window size: 25ms -> 16000 * 0.025 = 400
    # Hop length: 10ms -> 16000 * 0.010 = 160
    N_FFT = 400
    HOP_LENGTH = 160
    N_MELS = 64

    # SpecAugment Parameters
    FREQ_MASK_PARAM = 10  # Conservative masking (<20% of 64 mels)
    TIME_MASK_PARAM = 20  # Conservative masking relative to time steps

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True
    NUM_CLASSES = 12
    # Target labels mapping
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
    BATCH_SIZE = 128
    EPOCHS = 15
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    LABEL_SMOOTHING = 0.1

    # Early Stopping
    PATIENCE = 5

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # ==========================================
    # 5. Debugging
    # ==========================================
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 500
