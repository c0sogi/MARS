import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # General Setup
    # -------------------------------------------------------------------------
    PROJECT_NAME = "idea_16"
    SEED = 42
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Compute
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 8  # Optimized for 12 vCPUs

    # -------------------------------------------------------------------------
    # Data Paths
    # -------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Background noise directory for mixing
    NOISE_DIR = os.path.join(TRAIN_AUDIO_DIR, "_background_noise_")

    # -------------------------------------------------------------------------
    # Audio Parameters
    # -------------------------------------------------------------------------
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

    # -------------------------------------------------------------------------
    # Multi-Resolution Spectrogram Parameters
    # -------------------------------------------------------------------------
    # Window sizes for ~16ms, ~32ms, ~64ms at 16kHz (Powers of 2 for efficiency)
    # Cite Lesson 20: Multi-Resolution Spectrogram Stacking
    N_FFT_LIST = [256, 512, 1024]

    # Hop lengths (Fixed to ensure temporal alignment for stacking)
    # 128 samples = 8ms -> ~126 frames for 1s audio
    HOP_LENGTH_LIST = [128, 128, 128]

    # Mel Scale
    N_MELS = 128  # Native resolution
    F_MIN = 20
    F_MAX = 8000  # Nyquist

    # Input Image Size for Backbone (Freq, Time)
    # REMOVED: We use native resolution (128, ~126)
    # IMG_SIZE = (224, 224)

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------
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
    SILENCE_LABEL = "silence"
    UNKNOWN_LABEL = "unknown"

    # The full list of 12 classes
    ALL_LABELS = TARGET_LABELS + [SILENCE_LABEL, UNKNOWN_LABEL]
    NUM_CLASSES = len(ALL_LABELS)

    # Mappings
    LABEL2ID = {label: i for i, label in enumerate(ALL_LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(ALL_LABELS)}

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    MODEL_NAME = "resnest50d"
    PRETRAINED = True
    IN_CHANNELS = 3  # RGB channels correspond to the 3 resolutions

    # RNN / Head Configuration
    RNN_HIDDEN_SIZE = 128
    RNN_LAYERS = 2
    RNN_DROPOUT = 0.3
    BIDIRECTIONAL = True

    # Attention Pooling
    ATTN_NUM_HEADS = 4

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Debugging flags to control dataset size
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 200  # Number of samples to use if DEBUG is True

    EPOCHS = 25
    BATCH_SIZE = 256  # Increased due to smaller input size (16k vs 50k pixels)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Early Stopping
    PATIENCE = 6

    # -------------------------------------------------------------------------
    # Augmentation (GPU Native)
    # -------------------------------------------------------------------------
    # Probabilities
    AUG_PROB = 0.7
    NOISE_PROB = 0.5

    # SpecAugment Limits
    MASK_TIME_PROB = 0.2
    MASK_TIME_LIMIT = 20  # frames
    MASK_FREQ_PROB = 0.2
    MASK_FREQ_LIMIT = 20  # bands

    # Physics-based (Pitch/Time)
    PITCH_SHIFT_SEMITONES = 2.0
    TIME_STRETCH_RATE = 0.1

    @classmethod
    def setup(cls):
        """Creates necessary working and submission directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
