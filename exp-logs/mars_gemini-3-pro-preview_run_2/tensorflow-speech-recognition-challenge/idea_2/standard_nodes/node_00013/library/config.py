import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # 2. Audio Processing Parameters
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Spectrogram Parameters for High Resolution
    # Hop length of 160 (10ms) provides ~100 time steps for 1s audio
    # n_fft of 400 corresponds to 25ms window
    N_FFT = 400
    HOP_LENGTH = 160
    WIN_LENGTH = 400
    N_MELS = 128  # Frequency resolution

    # SpecAugment Parameters (Conservative <20%)
    FREQ_MASK_PARAM = 20  # Max frequency bands to mask
    TIME_MASK_PARAM = 20  # Max time steps to mask

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 12
    # Labels: yes, no, up, down, left, right, on, off, stop, go, silence, unknown
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
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Scheduler
    T_MAX = NUM_EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
