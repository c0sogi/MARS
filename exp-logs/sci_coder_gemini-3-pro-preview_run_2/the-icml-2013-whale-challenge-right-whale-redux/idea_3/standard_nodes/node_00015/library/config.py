import os
import torch


class Config:
    """
    Central configuration for Right Whale Call Detection.
    Includes file paths, audio processing parameters, model settings,
    and training hyperparameters.
    """

    # ==========================================
    # Paths
    # ==========================================
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Working directory for caching intermediate files (Idea 3 specific)
    WORKING_DIR = "./working/idea_3"

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # Audio Processing Parameters
    # ==========================================
    # Based on data analysis and proposed strategy
    SAMPLE_RATE = 2000
    DURATION = 2.0  # seconds

    # Spectrogram generation (High Resolution Strategy)
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 10
    FMAX = SAMPLE_RATE // 2  # Nyquist frequency (1000 Hz)

    # ==========================================
    # Model Parameters
    # ==========================================
    MODEL_NAME = "convnext_tiny"
    PRETRAINED = True
    IN_CHANNELS = 1  # Spectrogram is 1 channel
    NUM_CLASSES = 1  # Binary classification
    USE_GEM_POOLING = True  # Strategy: Generalized Mean Pooling

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42

    # Strategy: Maximize batch size for WeightedRandomSampler stability
    BATCH_SIZE = 128

    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler settings (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Hardware
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================
    # Debug / Development
    # ==========================================
    # Set DEBUG to True to run on a small subset of data for testing pipeline
    DEBUG = False
    DEBUG_SUBSET_SIZE = 500
