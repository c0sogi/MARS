import os


class Config:
    """
    Configuration for the Hybrid CRNN (Dilated EfficientNet-B2 + Bi-Directional GRU)
    Speech Command Recognition System.
    """

    # -------------------------------------------------------------------------
    # General & Hardware
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda"  # Assumes GPU availability as per task description

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directory for Caching and Checkpoints
    # Using 'idea_10' as the specific iteration identifier
    WORKING_DIR = "./working/idea_10"
    os.makedirs(WORKING_DIR, exist_ok=True)

    CACHE_DIR = WORKING_DIR
    CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Audio Signal Processing
    # -------------------------------------------------------------------------
    # High-fidelity settings for CRNN
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds

    # Spectrogram Parameters
    # N_FFT=1024 (64ms) and HOP_LENGTH=160 (10ms) provide high temporal resolution
    # essential for the GRU to model phonetic sequences.
    N_FFT = 1024
    HOP_LENGTH = 160
    N_MELS = 128
    F_MIN = 0
    F_MAX = None  # Defaults to Nyquist (8000 Hz)

    # -------------------------------------------------------------------------
    # Label Definitions (Fine-Grained Classification)
    # -------------------------------------------------------------------------
    # The 10 Target Commands required for the competition
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

    # Auxiliary Commands (Standard Speech Commands Set)
    # Used to maintain variance and prevent "Unknown" class collapse during training.
    AUX_LABELS = [
        "bed",
        "bird",
        "cat",
        "dog",
        "eight",
        "five",
        "four",
        "happy",
        "house",
        "marvin",
        "nine",
        "one",
        "seven",
        "sheila",
        "six",
        "three",
        "tree",
        "two",
        "wow",
        "zero",
    ]

    SILENCE_LABEL = "silence"
    UNKNOWN_LABEL = "unknown"  # Placeholder for mapping, not a training class

    # Full Class List for the Model Head (31 Classes)
    # Sorting ensures deterministic index mapping across runs.
    ALL_LABELS = sorted(TARGET_LABELS + AUX_LABELS) + [SILENCE_LABEL]
    NUM_CLASSES = len(ALL_LABELS)

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Backbone
    BACKBONE = "efficientnet_b2"
    PRETRAINED = True
    IN_CHANNELS = 1  # Mono audio converted to spectrogram

    # Sequential Head (CRNN)
    HIDDEN_DIM = 256
    GRU_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.3

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    EPOCHS = 50
    BATCH_SIZE = 32

    # Optimizer & Scheduler
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5  # Linear warmup to stabilize GRU weights
    MIN_LR = 1e-6

    # Regularization
    MIXUP_ALPHA = 1.0  # Strong mixup

    # Augmentation Parameters
    NOISE_SNR_MIN = 10
    NOISE_SNR_MAX = 30
    NOISE_PROB = 0.5

    # SpecAugment
    FREQ_MASK_PARAM = 20
    TIME_MASK_PARAM = 30
