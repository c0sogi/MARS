import os
import torch
import random
import numpy as np


class Config:
    # ==========================================
    # Project Structure & Paths
    # ==========================================
    PROJECT_NAME = "idea_3"
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Output Directories
    WORKING_DIR = f"./working/{PROJECT_NAME}"
    SUBMISSION_DIR = "./submission"

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Data Paths
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Audio File Directories (Relative to INPUT_ROOT)
    TRAIN_CURATED_DIR = "train_curated"
    TRAIN_NOISY_DIR = "train_noisy"
    TEST_DIR = "test"

    # ==========================================
    # Audio Parameters
    # ==========================================
    SR = 32000  # Sampling Rate (Hz)
    DURATION = 5  # Duration for training crops (seconds)
    N_MELS = 128  # Number of Mel bands
    N_FFT = 1024  # FFT window size
    HOP_LENGTH = 320  # Hop length
    FMIN = 20  # Minimum frequency
    FMAX = 16000  # Maximum frequency (Nyquist)

    # ==========================================
    # Model Architecture
    # ==========================================
    MODEL_NAME = "efficientnet_b2"
    NUM_CLASSES = 80
    PRETRAINED = True

    # CRNN Specifics
    USE_GRU = True
    GRU_HIDDEN_SIZE = 256
    GRU_LAYERS = 2
    BIDIRECTIONAL = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32  # Adjust based on GPU memory (A100 40GB allows larger batches)
    EPOCHS = 25  # Total training epochs
    LEARNING_RATE = 1e-3  # Initial learning rate
    WEIGHT_DECAY = 1e-2  # For AdamW
    ETA_MIN = 1e-6  # Minimum LR for Cosine Annealing

    # Augmentation
    MIXUP = True
    MIXUP_ALPHA = 0.4
    SPEC_AUG_TIME_MASK = 30
    SPEC_AUG_FREQ_MASK = 20

    # ==========================================
    # Compute & Debugging
    # ==========================================
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debug flag to run on a small subset of data
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    @staticmethod
    def set_seed(seed=42):
        """Sets the seed for reproducibility across all libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)


# Apply seed immediately upon import
Config.set_seed(Config.SEED)
