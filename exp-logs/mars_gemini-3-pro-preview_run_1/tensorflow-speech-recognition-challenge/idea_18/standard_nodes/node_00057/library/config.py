import os


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")
    NOISE_DIR = os.path.join(TRAIN_AUDIO_DIR, "_background_noise_")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for checkpoints, cache, and submissions
    WORK_DIR = "./working/idea_18"
    CACHE_DIR = WORK_DIR  # Using the work dir as cache root

    # Ensure working directory exists
    os.makedirs(WORK_DIR, exist_ok=True)

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    SR = 16000  # Sample Rate
    DURATION = 1.0  # Duration in seconds
    N_MELS = 128  # Number of Mel bands
    N_FFT = 1024  # FFT window size (64ms)
    HOP_LENGTH = 160  # Hop length (10ms)
    FMIN = 20  # Min frequency
    FMAX = SR // 2  # Max frequency
    TOP_DB = 80.0  # Top decibel for log conversion

    # ==========================================
    # Label Configuration
    # ==========================================
    # The 10 specific commands we must predict
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

    # Total expected classes for the fine-grained classifier:
    # 10 Targets + 20 Auxiliaries + 1 Silence = 31
    # The exact list of auxiliary classes is determined dynamically from the data
    NUM_CLASSES = 31

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    BACKBONE = "tf_efficientnet_b2_ns"
    PRETRAINED = True
    IN_CHANNELS = 1  # Spectrogram input is 1 channel
    DROP_RATE = 0.5  # Dropout rate
    USE_MULTI_SAMPLE_DROPOUT = True
    MULTI_SAMPLE_DROPOUT_COUNT = 8

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 128  # Optimized for A100 GPU
    NUM_WORKERS = 4

    # Optimizer & Scheduler
    LR = 1e-3  # Initial Learning Rate
    MIN_LR = 5e-5  # Floor LR for Cosine Annealing (0.05 * 1e-3)
    WEIGHT_DECAY = 1e-2

    # Epochs & SWA
    EPOCHS = 50  # Total training epochs
    SWA_START_EPOCH = 36  # Start SWA at epoch 36 (Phase 2)

    # ==========================================
    # Augmentation Parameters
    # ==========================================
    MIXUP_ALPHA = 1.0  # Alpha for Mixup distribution
    MIXUP_PROB = 0.5  # Probability of applying Mixup

    # Noise Injection
    NOISE_SNR_MIN = 10  # Min SNR (dB) for noise injection
    NOISE_SNR_MAX = 30  # Max SNR (dB) for noise injection
    NOISE_PROB = 0.8  # Probability of injecting background noise

    # SpecAugment
    TIME_MASK_PARAM = 10
    FREQ_MASK_PARAM = 10
    MASK_PROB = 0.5

    # ==========================================
    # Data Balancing Parameters
    # ==========================================
    # Target count for upsampling the 10 target commands
    TARGET_SAMPLES_PER_CLASS = 2000
