import os
import torch


class Config:
    # ==========================================
    # Project & Paths
    # ==========================================
    PROJECT_NAME = "RightWhaleDetection_Idea16"

    # Input Paths (Read-Only)
    INPUT_ROOT = "./input"
    TRAIN_CSV = "./metadata/train.csv"
    VAL_CSV = "./metadata/val.csv"
    TEST_CSV = "./metadata/test.csv"

    # Output Paths
    # We use a specific directory for this idea's artifacts
    OUTPUT_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Audio Processing
    # ==========================================
    SAMPLE_RATE = 2000
    DURATION = 2.0  # Seconds

    # Spectrogram Parameters
    # High frequency resolution (1024 FFT @ 2000Hz -> ~2Hz per bin)
    N_FFT = 1024
    HOP_LENGTH = 32
    N_MELS = 128  # Log-Mel resolution
    FMIN = 15  # Right whale calls are low frequency
    FMAX = 1000  # Nyquist limit

    # Calculated dimensions
    # Time steps = (2000 * 2.0) / 32 = 125 frames

    # ==========================================
    # Data Augmentation
    # ==========================================
    MIXUP_ALPHA = 0.4

    # SpecAugment Constraints
    # Constraint: Time Mask < 200ms
    # Frame duration = 32 / 2000 = 0.016s
    # Max frames = 0.2s / 0.016s = 12.5 frames
    TIME_MASK_PARAM = 10  # Set conservatively below 12
    FREQ_MASK_PARAM = 16

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "ContextGatedResNet18"
    BACKBONE = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 1
    DROPOUT = 0.2

    # Ensemble Strategy
    SEEDS = [42, 101, 202, 303, 404]

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    EPOCHS = 25
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Loss Function
    # Explicitly handle 1:9 imbalance
    POS_WEIGHT = 9.0

    # Scheduler & Early Stopping
    PATIENCE = 6
    FACTOR = 0.5  # ReduceLROnPlateau factor

    # ==========================================
    # Hardware / System
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Create necessary output directories."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on import
Config.setup()
