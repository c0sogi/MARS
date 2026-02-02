import os
import random
import numpy as np
import torch


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    """
    Global configuration for the Lung Function Decline prediction pipeline.
    """

    # ==========================
    # General Settings
    # ==========================
    PROJECT_NAME = "osic-pulmonary-fibrosis-progression"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLES = 50
    NUM_WORKERS = 4  # 12 vCPUs available

    # ==========================
    # File Paths
    # ==========================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Metadata files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Image directories (Relative to INPUT_DIR as per metadata)
    TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
    TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

    # Caching
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
    LOAD_CACHED_DATA = True

    # Output
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # ==========================
    # Image Processing (MIP)
    # ==========================
    IMG_SIZE = 224
    SLICES_PERCENTAGE = 0.5  # Use middle 50% of slices
    HU_MIN = -1000  # Hounsfield Unit lower bound
    HU_MAX = 400  # Hounsfield Unit upper bound

    # ==========================
    # Model Hyperparameters
    # ==========================
    BACKBONE = "resnet18"
    PRETRAINED = True

    # Tabular features: Age (1) + Sex (2) + SmokingStatus (3) + Baseline_Percent (1)
    # Sex: Male, Female (2)
    # Smoking: Ex-smoker, Never smoked, Currently smokes (3)
    N_TABULAR_FEATURES = 7
    HIDDEN_DIM = 512

    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-2

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # Augmentation
    USE_AUGMENTATION = True

    # ==========================
    # Metric / Loss
    # ==========================
    # Constants for the modified Laplace Log Likelihood
    MAX_ERROR = 1000
    MIN_SIGMA = 70

    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Ensure necessary directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
