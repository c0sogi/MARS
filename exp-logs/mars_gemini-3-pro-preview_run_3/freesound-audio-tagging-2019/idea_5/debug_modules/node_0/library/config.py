import os
import torch


class Config:
    # ==========================================
    # General Setup
    # ==========================================
    PROJECT_NAME = "audio_tagging_idea_5"
    SEED = 42
    NUM_WORKERS = 4  # Adjust based on vCPUs (12 available)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # Data Paths
    # ==========================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files (generated previously)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directories
    WORKING_DIR = "./working/idea_5"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio Processing
    # ==========================================
    SAMPLE_RATE = 32000
    DURATION = 30  # seconds
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512
    F_MIN = 20
    F_MAX = 16000  # Nyquist for 32kHz

    # Calculated Input Size: (128, 1876) for 30s at 32kHz with hop 512
    # 30 * 32000 / 512 = 1875 frames approx.

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    NUM_CLASSES = 80
    PRETRAINED = True
    IN_CHANNELS = 1  # Single channel input (summed weights)

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 100

    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_LR = 1e-3  # For OneCycleLR

    # Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 0.4
    SPEC_AUG_TIME_MASK = 30
    SPEC_AUG_FREQ_MASK = 20

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    @classmethod
    def setup(cls):
        """Ensures all necessary directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
