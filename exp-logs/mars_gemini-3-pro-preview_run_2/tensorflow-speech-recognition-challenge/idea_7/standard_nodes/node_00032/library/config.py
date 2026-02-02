import os
import torch


class Config:
    """
    Configuration class for the Speech Command Recognition task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    PROJECT_ROOT = "."
    INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
    METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")

    # Working directory for caching and checkpoints
    WORKING_DIR = os.path.join(PROJECT_ROOT, "working", "idea_9")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Submission
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure submission directory exists
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # 2. Audio Processing Parameters
    # ==========================================
    # Strategy: High-Fidelity Spectrograms
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds

    # Spectrogram extraction
    N_MELS = 128  # High frequency resolution
    N_FFT = 400  # 25ms window at 16kHz
    HOP_LENGTH = 160  # 10ms hop at 16kHz (High temporal resolution)
    F_MIN = 0
    F_MAX = None  # Defaults to SR // 2

    # ==========================================
    # 3. Label Configuration
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

    # Mappings
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}

    # ==========================================
    # 4. Model Architecture
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True
    IN_CHANNELS = 1
    # Specific architectural tweaks defined in model file, controlled here if needed
    DROPOUT = 0.2

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    SEED = 42

    # Optimization
    BATCH_SIZE = 128  # Adjusted for A100 (40GB)
    NUM_EPOCHS = 25  # Max epochs within 24h constraint
    LEARNING_RATE = 1e-3  # Initial LR for AdamW
    WEIGHT_DECAY = 1e-2

    # Regularization
    LABEL_SMOOTHING = 0.1  # As per strategy

    # Scheduler (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-6

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # Gradient Clipping
    MAX_GRAD_NORM = 1.0

    # ==========================================
    # 6. Data Augmentation (SpecAugment)
    # ==========================================
    # Conservative masking to preserve signal
    FREQ_MASK_PARAM = 20
    TIME_MASK_PARAM = 20

    # ==========================================
    # 7. Hardware & Computation
    # ==========================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Based on 12 vCPUs available
    PIN_MEMORY = True

    # ==========================================
    # 8. Debugging
    # ==========================================
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 1000
