import os
import torch


class Config:
    """
    Configuration class for the Siamese Feature-Difference Network pipeline.
    Centralizes all hyperparameters, file paths, and system settings.
    """

    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4
    # Automatically detect device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # File Paths & Directories
    # ==========================================
    # Input directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs (Already generated)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for artifacts (Checkpoints, Cache)
    # Using 'idea_6' as the specific experiment folder
    WORK_DIR = "./working/idea_6"
    os.makedirs(WORK_DIR, exist_ok=True)

    # Output path for the best model checkpoint
    BEST_MODEL_PATH = os.path.join(WORK_DIR, "best_model.pth")

    # Submission directory and file
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Preprocessing & Dimensions
    # ==========================================
    # Original spectrogram dimensions: (6 positions, 273 frequency bins, 256 time steps)
    ORIG_SHAPE = (6, 273, 256)

    # Target dimensions for the model
    # We pad height 273 -> 288 to be a multiple of 32 (compatible with EfficientNet)
    # Width remains 256
    IMG_HEIGHT = 288
    IMG_WIDTH = 256

    # ==========================================
    # Model Architecture
    # ==========================================
    BACKBONE = "efficientnet_b0"
    PRETRAINED = True

    # The model accepts two streams (On-Target, Off-Target)
    # Each stream is a stack of 3 observations (e.g., A1, A2, A3)
    IN_CHANNELS = 3

    # Binary classification
    NUM_CLASSES = 1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    EPOCHS = 15
    BATCH_SIZE = 64

    # Optimizer (AdamW)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (CosineAnnealingLR)
    T_MAX = 12
    ETA_MIN = 1e-6

    # Early Stopping
    PATIENCE = 5

    # ==========================================
    # Augmentation & Regularization
    # ==========================================
    # Mixup alpha parameter
    MIXUP_ALPHA = 0.2

    # ==========================================
    # Debugging & Development
    # ==========================================
    # Set to True to run on a small subset of data for testing the pipeline
    DEBUG = False

    # Number of samples to use in debug mode
    DEBUG_SAMPLE_SIZE = 200
