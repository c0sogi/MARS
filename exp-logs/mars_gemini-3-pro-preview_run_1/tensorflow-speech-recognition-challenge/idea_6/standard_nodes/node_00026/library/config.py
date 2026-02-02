import os
import torch


class PathConfig:
    """
    Defines file paths for input data, metadata, and working directories.
    """

    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

    # Metadata paths (pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
    VAL_META = os.path.join(METADATA_DIR, "val.csv")
    TEST_META = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for Idea 7 (Dilated-EB2)
    WORKING_DIR = "./working/idea_7"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Output paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Cache paths for processed data
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_balanced.parquet")


class AudioConfig:
    """
    Defines parameters for audio signal processing and spectrogram generation.
    High-fidelity settings for FPN-EB2.
    """

    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Spectrogram Parameters
    # 128 Mels for high spectral resolution
    # 1024 FFT / 160 Hop for high temporal resolution (approx 10ms hop)
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 160
    F_MIN = 0
    F_MAX = None  # Defaults to SAMPLE_RATE / 2


class ModelConfig:
    """
    Defines model architecture hyperparameters.
    """

    MODEL_NAME = "efficientnet_b2"
    PRETRAINED = True
    IN_CHANNELS = 1
    NUM_CLASSES = 12

    # Head settings
    DROPOUT = 0.3


class TrainConfig:
    """
    Defines training hyperparameters and regularization strategies.
    """

    SEED = 42
    DEBUG = False  # Set to True to run on a small subset

    # Training Loop
    BATCH_SIZE = 32
    EPOCHS = 50
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    ETA_MIN = 1e-6

    # Regularization
    # Strong Mixup as per strategy
    MIXUP_ALPHA = 1.0
    USE_SPECAUGMENT = True

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10


class DataConfig:
    """
    Defines label mappings and balancing strategies.
    """

    # The 10 specific target commands
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

    # Full label list for classification (12 classes)
    # Order matters: targets first, then special classes
    ALL_LABELS = TARGET_LABELS + ["silence", "unknown"]

    # Mappings
    LABEL2ID = {label: i for i, label in enumerate(ALL_LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(ALL_LABELS)}

    # Balancing Strategy
    # We want to upsample targets and downsample unknown to create a balanced batch
    # This is often handled by a WeightedRandomSampler or custom dataset logic
    TARGET_SAMPLE_WEIGHT = 1.0
    UNKNOWN_SAMPLE_WEIGHT = 0.3  # Downsample unknown class
    SILENCE_SAMPLE_WEIGHT = 0.5  # Silence is synthetic, control its prevalence
