import os
import torch


class Config:
    # -----------------------
    # General Settings
    # -----------------------
    SEED = 42
    DEBUG = False
    NUM_WORKERS = 12  # Utilizing available vCPUs

    # -----------------------
    # Audio Processing
    # -----------------------
    SAMPLE_RATE = 2000  # From data analysis
    DURATION = 2.0  # Max duration in dataset
    N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 4000 samples

    # Spectrogram Generation
    N_MELS = 224  # High resolution for ConvNeXt
    N_FFT = 1024  # Window size for FFT
    HOP_LENGTH = 18  # Results in ~222 frames for 4000 samples
    FMIN = 20  # Min frequency
    FMAX = 1000  # Nyquist frequency (SR/2)

    # -----------------------
    # Model Architecture
    # -----------------------
    BACKBONE = "convnext_tiny"
    PRETRAINED = True
    IN_CHANS = 1  # Log-Mel Spectrogram has 1 channel
    NUM_CLASSES = 1  # Binary classification (Whale vs Noise)
    USE_GEM = True  # Generalized Mean Pooling

    # -----------------------
    # Training Hyperparameters
    # -----------------------
    EPOCHS = 20
    BATCH_SIZE = 128  # Large batch size for A100
    LEARNING_RATE = 1e-3
    MIN_LR = 1e-6
    WEIGHT_DECAY = 0.01

    # Regularization & Augmentation
    MIXUP_ALPHA = 0.4
    USE_SPECAUGMENT = True
    USE_CLASS_WEIGHTS = True  # Weight loss by inverse class frequency

    # -----------------------
    # Paths & Directories
    # -----------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working Directories
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_4")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Output Files
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -----------------------
    # Compute
    # -----------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup(cls):
        """Ensures that working and cache directories exist."""
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
