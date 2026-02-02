import os
import torch


class Config:
    """
    Global configuration for the Right Whale Detection task.
    Implements settings for the 'Ensemble of Time-Preserving Context-Gated
    Hierarchical ConvNeXt-Pico CRNNs' solution.
    """

    # ==========================================
    # Project & Path Configuration
    # ==========================================
    PROJECT_NAME = "idea_17"
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio & Spectrogram Configuration
    # ==========================================
    SAMPLE_RATE = 2000
    DURATION = 2.0  # seconds
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)  # 4000 samples

    # Mel Spectrogram Parameters
    # N_FFT = 1024 (Large window for frequency resolution)
    # HOP_LENGTH = 20 (Small hop to maintain temporal density: 4000/20 = 200 frames)
    N_FFT = 1024
    WIN_LENGTH = 1024
    HOP_LENGTH = 20
    N_MELS = 128
    F_MIN = 10.0
    F_MAX = 1000.0  # Nyquist for 2000Hz SR
    POWER = 2.0

    # ==========================================
    # Model Architecture Configuration
    # ==========================================
    BACKBONE = "convnext_pico"
    PRETRAINED = True
    IN_CHANNELS = 1
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Ensemble Strategy: 10 independent models
    SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]

    BATCH_SIZE = 256
    EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2  # Standard for AdamW
    PATIENCE = 6  # Early stopping patience

    # Loss Function
    # Explicitly handle 1:9 imbalance
    POS_WEIGHT = 9.0

    # ==========================================
    # Augmentation Settings
    # ==========================================
    MIXUP_ALPHA = 0.4

    # SpecAugment
    # Time mask constraint: < 200ms.
    # 200ms = 0.2s * 2000Hz = 400 samples.
    # Max frames = 400 / HOP_LENGTH(20) = 20 frames.
    TIME_MASK_PARAM = 20
    FREQ_MASK_PARAM = 20

    # ==========================================
    # Compute & Debugging
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug flags to speed up development cycles if needed
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500

    @classmethod
    def get_cache_path(cls, filename):
        """
        Generates a full path for saving/loading cached files
        within the project's working directory.
        """
        return os.path.join(cls.WORKING_DIR, filename)
