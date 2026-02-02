import os
import random
import numpy as np
import torch


class Config:
    # =========================================================================
    # General Setup
    # =========================================================================
    SEED = 42
    NUM_WORKERS = 2  # Conservative for the available vCPUs
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # File Paths & Directories
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Specific working directory for this idea (idea_20)
    # This is used for caching processed numpy arrays
    WORK_DIR = "./working/idea_20/"

    # Submission output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Files
    TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
    TEST_JSON = os.path.join(INPUT_DIR, "test.json")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    # Image Dimensions
    ORIG_IMG_SIZE = 75
    IMG_SIZE = 224  # Upsampled size for ResNet

    # Normalization Constants (Derived from Data Analysis)
    # Band 1 (HH)
    BAND1_MIN = -45.5944
    BAND1_MAX = 32.1806
    # Band 2 (HV)
    BAND2_MIN = -45.6555
    BAND2_MAX = 17.8628

    # =========================================================================
    # Model Architecture
    # =========================================================================
    MODEL_NAME = "resnet18"
    PRETRAINED = True
    NUM_CLASSES = 1
    DROPOUT_RATE = 0.5

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.01

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 10

    # Loss
    LABEL_SMOOTHING = 0.05

    # Training Loop
    N_FOLDS = 5
    MAX_EPOCHS = 100  # Upper bound, controlled by early stopping
    EARLY_STOPPING_PATIENCE = 15

    # SWA (Stochastic Weight Averaging)
    SWA_EPOCHS = 12
    SWA_LR = 1e-4  # Constant LR for SWA phase

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist.
        """
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
