import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea (caching, checkpoints)
    WORK_DIR = "./working/idea_12"
    os.makedirs(WORK_DIR, exist_ok=True)

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Submission output
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # 2. Audio Processing (High-Fidelity)
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Spectrogram Parameters
    N_MELS = 128
    N_FFT = 1024  # 64ms window
    HOP_LENGTH = 160  # 10ms hop
    F_MIN = 0
    F_MAX = None  # Nyquist

    # Augmentation
    NOISE_SNR_MIN = 10
    NOISE_SNR_MAX = 30
    NOISE_PROB = 0.5

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    BACKBONE = "tf_efficientnet_b2"
    PRETRAINED = True
    IN_CHANNELS = 1  # Log-Mel Spectrogram is 1 channel

    # Energy Gating
    USE_ENERGY_GATING = True

    # Classification Heads
    # The model trains on fine-grained classes (all words)
    # and maps to final 12 classes during inference.
    # We assume approx 35 classes in standard Speech Commands dataset.
    # The exact number will be determined by the Dataset class.

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 128  # A100 has 40GB, can handle large batches

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Regularization
    MIXUP_ALPHA = 1.0
    DROPOUT_RATE = 0.2
    DROP_PATH_RATE = 0.1

    # Compute
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 5. Labels and Mapping
    # ==========================================
    # The 10 specific commands we need to predict
    TARGET_LABELS = {
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
    }

    # The full set of 12 output labels for the competition metric
    OUTPUT_LABELS = [
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

    # Label to Index mapping for the final submission
    LABEL2ID = {label: i for i, label in enumerate(OUTPUT_LABELS)}
    ID2LABEL = {i: label for label, i in LABEL2ID.items()}

    # ==========================================
    # 6. Debug / Development
    # ==========================================
    # Set to True to run on a small subset for testing pipeline
    DEBUG = False
    DEBUG_SUBSET_SIZE = 0.05  # Fraction of data to use in debug mode


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
