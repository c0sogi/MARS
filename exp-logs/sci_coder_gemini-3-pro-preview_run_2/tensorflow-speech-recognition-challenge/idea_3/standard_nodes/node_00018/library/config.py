import os
import torch


class Config:
    """
    Central configuration for the Speech Command Recognition task.
    Implements settings for ConvNeXt-Tiny model, high-resolution spectrograms,
    and robust training strategies.
    """

    # ==========================================
    # 1. Paths and Directories
    # ==========================================
    PROJECT_ROOT = "."
    INPUT_ROOT = os.path.join(PROJECT_ROOT, "input")
    METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")

    # Working directory for Idea 4 (Cache and Checkpoints)
    WORKING_DIR = os.path.join(PROJECT_ROOT, "working", "idea_4")
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Submission directory
    SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "submission")
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Model Checkpoint Path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================================
    # 2. Audio Processing Parameters
    # ==========================================
    # Standard sample rate for speech command datasets
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    N_SAMPLES = int(SAMPLE_RATE * DURATION)

    # High-Resolution Log-Mel Spectrogram Settings
    # Hop length of 160 at 16kHz corresponds to 10ms stride
    N_FFT = 1024
    HOP_LENGTH = 160
    N_MELS = 128
    F_MIN = 20
    F_MAX = 8000

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    # Backbone: ConvNeXt-Tiny (modernized CNN)
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True

    # Input channels: 1 (Spectrogram) -> requires weight averaging of RGB weights
    IN_CHANNELS = 1

    # Head: Attention Pooling to handle temporal variance and silence
    USE_ATTENTION_POOLING = True

    # Target Labels
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
    # 4. Training Hyperparameters
    # ==========================================
    SEED = 42

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Optimization
    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.01  # Standard for AdamW

    # Regularization
    LABEL_SMOOTHING = 0.1

    # SpecAugment Parameters (Calibrated to be <20% of dimensions)
    # Time frames approx 100 -> Mask 20
    # Freq bins 128 -> Mask 20
    SPEC_AUG_TIME_MASK = 20
    SPEC_AUG_FREQ_MASK = 20

    # ==========================================
    # 5. Debugging and Development
    # ==========================================
    # Set DEBUG to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 2000
