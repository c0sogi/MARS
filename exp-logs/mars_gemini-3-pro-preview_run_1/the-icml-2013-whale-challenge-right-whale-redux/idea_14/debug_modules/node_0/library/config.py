import os
import torch


class Config:
    """
    Configuration module for the Right Whale Detection task.
    Centralizes all hyperparameters, constants, and path definitions.
    """

    # =========================================================================
    # Directories & Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and model checkpoints
    WORKING_DIR = "./working/idea_14"

    # Output directory for submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata File Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Ensure essential directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Audio Processing Settings
    # =========================================================================
    # Fixed parameters as per task description
    SAMPLE_RATE = 2000
    DURATION = 2.0  # seconds

    # Spectrogram generation
    N_FFT = 1024  # High frequency resolution
    HOP_LENGTH = 512  # Standard overlap
    N_MELS = 128

    # Derived constants
    NUM_SAMPLES = int(SAMPLE_RATE * DURATION)  # 4000 samples

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 30

    # Imbalance Handling
    POS_WEIGHT = 9.0  # Weight for the positive (whale) class

    # Optimization & Scheduling
    PATIENCE = 7  # Early stopping patience
    LR_FACTOR = 0.5  # ReduceLROnPlateau factor
    LR_PATIENCE = 3  # ReduceLROnPlateau patience
    MIN_LR = 1e-6

    # =========================================================================
    # Augmentation & Regularization
    # =========================================================================
    MIXUP_ALPHA = 0.4

    # SpecAugment
    # Constraint: Time Mask width < 200ms.
    # With SR=2000 and Hop=512, one frame is 256ms.
    # We set TIME_MASK_PARAM to 1 (range [0, 1)) to minimize/avoid masking full frames
    # while strictly adhering to the time constraint relative to the spectrogram resolution.
    FREQ_MASK_PARAM = 20
    TIME_MASK_PARAM = 1

    # =========================================================================
    # Ensemble Strategy
    # =========================================================================
    ENSEMBLE_SIZE = 5
    # Fixed seeds for reproducibility across ensemble members
    SEEDS = [42, 101, 202, 303, 404]

    # =========================================================================
    # System & Compute
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # Debugging flag (can be toggled to run on a subset)
    DEBUG = False
