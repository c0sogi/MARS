import os
import torch


class Config:
    # ==========================================
    # Paths & Directories
    # ==========================================
    PROJECT_ROOT = "."
    INPUT_ROOT = os.path.join(PROJECT_ROOT, "input")
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

    METADATA_DIR = os.path.join(PROJECT_ROOT, "metadata")
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for this specific idea (Idea 12)
    WORK_DIR = os.path.join(PROJECT_ROOT, "working", "idea_12")
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "submission", "submission.csv")

    # Ensure directories exist
    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Audio Configuration
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 16000

    # ==========================================
    # Feature Extraction (Multi-Res Spectrogram)
    # ==========================================
    # We use 3 resolutions mapped to RGB channels
    # Window sizes: 20ms, 40ms, 60ms
    # At 16kHz: 20ms=320, 40ms=640, 60ms=960
    N_FFT_RESOLUTIONS = [320, 640, 960]
    WIN_LENGTHS = [320, 640, 960]

    # Hop length must be consistent across channels to align time dimension
    # 10ms hop = 160 samples
    HOP_LENGTH = 160

    N_MELS = 80
    F_MIN = 20.0
    F_MAX = 8000.0  # Nyquist for 16kHz

    # Normalization stats (approximate from dataset analysis)
    # These will be applied after log conversion
    NORM_MEAN = -4.2677393
    NORM_STD = 4.5689974

    # ==========================================
    # Label Configuration
    # ==========================================
    # The 10 core commands
    COMMANDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

    # Full label list for classification (12 classes)
    LABELS = COMMANDS + ["silence", "unknown"]
    NUM_CLASSES = len(LABELS)

    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}

    # ==========================================
    # Model Architecture
    # ==========================================
    # Backbone
    BACKBONE_NAME = "skresnet34"  # Selective Kernel ResNet34
    PRETRAINED = True
    IN_CHANNELS = 3  # 3 resolutions

    # Neck & Head
    PROJECTION_DIM = 1024
    RNN_HIDDEN_SIZE = 256
    RNN_LAYERS = 2
    RNN_DROPOUT = 0.3
    ATTENTION_HEADS = 4

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # AdamW weight decay

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 5

    # ==========================================
    # Augmentation (SpecAugment)
    # ==========================================
    # Applied on the Log-Mel Spectrogram
    # Time mask < 20% of duration.
    # Duration is ~100 frames (16000/160). 20% is ~20 frames.
    TIME_MASK_PARAM = 20
    FREQ_MASK_PARAM = 20  # Mask up to 20 mel bands

    # Mixup/Cutmix are explicitly disabled based on Idea logic
    USE_MIXUP = False
