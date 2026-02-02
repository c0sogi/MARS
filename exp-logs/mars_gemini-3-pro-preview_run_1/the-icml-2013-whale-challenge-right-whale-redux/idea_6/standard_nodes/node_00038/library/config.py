import os
import torch


class Config:
    """
    Central configuration for the Resolution-Aligned Time-Preserving EfficientNet-BiGRU experiment.
    """

    # ==========================================
    # Paths and Directories
    # ==========================================
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    # Metadata is pre-generated in ./metadata
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching and model artifacts
    WORKING_DIR = "./working/idea_6"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    SR = 2000  # Sample Rate (Hz)
    DURATION = 2.0  # Clip duration (seconds)

    # Spectrogram Generation
    # Goal: ~200-250 time frames.
    # 2000 samples/sec * 2 sec = 4000 samples.
    # Hop Length 16 => 4000 / 16 = 250 frames.
    # 1 frame = 16 samples / 2000 Hz = 8 ms.
    N_FFT = 256  # Window size (~128ms)
    HOP_LENGTH = 16  # Hop size (~8ms)
    N_MELS = 128  # Frequency resolution
    FMIN = 10  # Min frequency
    FMAX = 1000  # Max frequency (Nyquist)

    # ==========================================
    # Model Architecture Parameters
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True

    # RNN Head
    RNN_HIDDEN_SIZE = 128
    RNN_LAYERS = 2
    RNN_DROPOUT = 0.3

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32  # Reduced to prevent OOM
    EPOCHS = 25  # With Early Stopping
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Class Imbalance Handling
    # Ratio is approx 1:9. Pos_weight > 1 increases recall.
    POS_WEIGHT = 9.0

    # Data Augmentation
    MIXUP_ALPHA = 0.4

    # SpecAugment
    # Time mask max width constraint: 200ms.
    # 200ms / 8ms per frame = 25 frames.
    TIME_MASK_PARAM = 20  # Safe upper bound
    FREQ_MASK_PARAM = 20

    # ==========================================
    # System & Debugging
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    # Debugging flags to control dataset size
    DEBUG = False
    DEBUG_SAMPLES = 500  # Number of samples to use if DEBUG is True
