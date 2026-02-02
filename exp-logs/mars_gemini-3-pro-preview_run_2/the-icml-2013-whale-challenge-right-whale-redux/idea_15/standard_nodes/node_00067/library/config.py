import os
import torch
import random
import numpy as np


class Config:
    """
    Global configuration for the Right Whale Call Detection task.
    Implements the 'Calibrated Heterogeneous Stacked Ensemble with Dynamic Range Correction'.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_ROOT = "./input"
    TRAIN_DIR = os.path.join(INPUT_ROOT, "train2")
    TEST_DIR = os.path.join(INPUT_ROOT, "test2")

    # Metadata (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output / Working Directory
    # Using idea_15 as specified in the prompt for this iteration
    WORKING_DIR = "./working/idea_15"

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # Audio Parameters ("Golden Recipe" + Dynamic Range Correction)
    # =========================================================================
    SR = 2000  # Sampling Rate (all files are 2kHz)
    N_FFT = 1024  # High frequency resolution
    HOP_LENGTH = 64  # High temporal resolution
    N_MELS = 128  # Mel bands
    FMIN = 0
    FMAX = None  # Defaults to SR/2
    POWER = 2.0  # Power spectrogram

    # Critical Correction: Clamps noise floor to prevent silence from skewing normalization
    TOP_DB = 80.0

    # Normalization: We use instance-wise Zero-Mean Unit-Variance in the dataset class,
    # so we do NOT normalize the mel-filterbank area itself.
    NORMALIZE_MELS = False

    # =========================================================================
    # Model Parameters (Heterogeneous Ensemble)
    # =========================================================================
    # 1. EfficientNet-B0 (Noisy Student, JFT-300M pre-trained)
    # 2. ResNet-34 (Standard ImageNet)
    MODEL_NAMES = ["tf_efficientnet_b0.ns_jft_in1k", "resnet34"]

    IN_CHANNELS = 1  # Adapted from RGB
    NUM_CLASSES = 1  # Binary Classification
    USE_GEM_POOLING = True  # Generalized Mean Pooling
    PRETRAINED = True

    # =========================================================================
    # Training Parameters
    # =========================================================================
    SEED = 42
    NUM_FOLDS = 5

    # Training Loop
    EPOCHS = 20
    BATCH_SIZE = 128  # Safe for A100 with these model sizes

    # Optimization
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Low weight decay to prevent over-constraining NS weights

    # Scheduler (Cosine Annealing)
    T_MAX = EPOCHS
    MIN_LR = 1e-6

    # Early Stopping
    # Structural Change: Monitor Loss for better probability calibration
    EARLY_STOPPING_PATIENCE = 5
    EARLY_STOPPING_MONITOR = "val_loss"
    EARLY_STOPPING_MODE = "min"

    # Compute
    NUM_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Debugging
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SUBSET_SIZE = 200  # Number of samples to use in debug mode


def set_seed(seed=42):
    """Sets the random seed for reproducibility across libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
