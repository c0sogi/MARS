import os
import torch


class Config:
    """
    Configuration class for the Speech Command Recognition project.
    Contains file paths, audio parameters, model hyperparameters, and label definitions.
    """

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_ROOT = "./input"
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    WORKING_DIR = "./working"
    # Directory for caching processed data (e.g., spectrograms)
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Audio Processing Parameters
    # -------------------------------------------------------------------------
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    N_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Mel Spectrogram Parameters
    # Resulting shape: [N_MELS, Time]
    # Time frames ~= N_SAMPLES / HOP_LENGTH. 16000/160 = 100 frames.
    # Cite solution_lesson_node_00004: Increased resolution to 128
    N_MELS = 128
    # Cite solution_lesson_node_00005: Increase N_FFT to 2048 to support N_MELS=128
    N_FFT = 2048
    HOP_LENGTH = 160

    # -------------------------------------------------------------------------
    # Labels
    # -------------------------------------------------------------------------
    # The order matches the metric description requirements.
    # 'unknown' handles all non-target words. 'silence' handles background noise.
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

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    # Class Balancing
    # The 'unknown' class is much larger than others. We downsample it during
    # training data creation to avoid bias.
    UNKNOWN_TRAIN_SAMPLE_COUNT = 2000

    # -------------------------------------------------------------------------
    # System / Compute
    # -------------------------------------------------------------------------
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of subprocesses for data loading
