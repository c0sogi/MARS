import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # 1. General Settings
    # ==========================================
    SEED = 42
    NUM_WORKERS = (
        0  # Set to 0 for GPU-resident pipeline to avoid multiprocessing overhead
    )
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # 2. Audio Parameters
    # ==========================================
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

    # Spectrogram generation parameters for GPU
    N_FFT = 1024  # Large FFT for spectral oversampling
    HOP_LENGTH = 160  # 10ms hop length at 16kHz
    WIN_LENGTH = 400  # 25ms window length at 16kHz
    N_MELS = 128  # High resolution Mel bands
    F_MIN = 0
    F_MAX = 8000  # Nyquist frequency

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    BACKBONE = "efficientnet_v2_b0"
    NUM_CLASSES = 12
    IN_CHANNELS = 1  # Single channel spectrogram input

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32  # Small batch size for high frequency gradient updates
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_EPOCHS = 40
    PATIENCE = 7  # Early stopping patience
    LABEL_SMOOTHING = 0.1

    # Augmentation Parameters
    NOISE_INJECTION_PROB = 0.8
    SPECAUG_TIME_MASK_PARAM = 20  # Max time steps to mask
    SPECAUG_FREQ_MASK_PARAM = 20  # Max freq bins to mask

    # ==========================================
    # 5. Labels
    # ==========================================
    # The 12 classes for the task
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
    # Mappings
    LABEL2ID = {label: i for i, label in enumerate(LABELS)}
    ID2LABEL = {i: label for i, label in enumerate(LABELS)}

    # The core command words (others map to unknown/silence in metadata)
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

    # ==========================================
    # 6. Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"
    SUBMISSION_DIR = "./submission"

    # Source Audio Directories
    TRAIN_AUDIO_DIR = os.path.join(INPUT_ROOT, "train", "audio")
    TEST_AUDIO_DIR = os.path.join(INPUT_ROOT, "test", "audio")

    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Paths
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Cache Paths (for GPU-resident data loading)
    CACHE_TRAIN_WAVEFORMS = os.path.join(WORKING_DIR, "train_waveforms.npy")
    CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")
    CACHE_VAL_WAVEFORMS = os.path.join(WORKING_DIR, "val_waveforms.npy")
    CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")
    CACHE_TEST_WAVEFORMS = os.path.join(WORKING_DIR, "test_waveforms.npy")
    CACHE_TEST_FNAMES = os.path.join(WORKING_DIR, "test_fnames.npy")
    CACHE_BACKGROUND_NOISE = os.path.join(WORKING_DIR, "background_noise.npy")

    @classmethod
    def setup(cls):
        """
        Initializes the environment: creates directories and sets random seeds.
        """
        # Create necessary directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior in CuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
