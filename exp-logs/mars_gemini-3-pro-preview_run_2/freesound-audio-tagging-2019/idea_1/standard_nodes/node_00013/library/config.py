import os
import random
import numpy as np
import torch


class Config:
    # ==== General Settings ====
    PROJECT_NAME = "idea_1"
    SEED = 42
    DEBUG = False  # Set to True to use a subset of data for debugging

    # ==== Paths ====
    # Input directories (Read-Only)
    INPUT_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output directories
    WORKING_DIR = os.path.join("./working", PROJECT_NAME)
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Checkpoint path
    BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==== Audio Parameters ====
    SAMPLE_RATE = 32000
    DURATION = 10  # Seconds
    N_MELS = 128
    N_FFT = 1024
    HOP_LENGTH = 320
    FMIN = 20
    FMAX = 16000  # Nyquist frequency for 32kHz

    # Calculated samples length
    AUDIO_LEN = int(SAMPLE_RATE * DURATION)

    # ==== Model Parameters ====
    NUM_CLASSES = 80
    IN_CHANNELS = 1  # Log-mel spectrogram is 1 channel

    # ==== Training Hyperparameters ====
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    MAX_EPOCHS = 50
    PATIENCE = 7  # Early stopping patience

    # ==== Augmentation Parameters ====
    SPEC_AUG_FREQ_MASK = 30
    SPEC_AUG_TIME_MASK = 100

    # Data Loading
    NUM_WORKERS = 4

    # Debugging / Dataset Control
    # If DEBUG is True, these limit the number of samples used
    MAX_TRAIN_SAMPLES = None
    MAX_VAL_SAMPLES = None

    @classmethod
    def setup(cls):
        """Creates necessary output directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def seed_everything(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Initialize directories when module is imported
Config.setup()
