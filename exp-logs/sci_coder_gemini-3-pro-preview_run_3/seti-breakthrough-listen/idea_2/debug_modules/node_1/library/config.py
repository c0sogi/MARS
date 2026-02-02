import os
import torch


class Config:
    # ==========================================
    # General Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 500  # Number of samples to use if DEBUG is True

    # ==========================================
    # File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output Directories
    WORKING_DIR = "./working"
    CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")
    SUBMISSION_DIR = "./submission"

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Parameters
    # ==========================================
    # Input dimensions
    # Original is (6, 273, 256). We resize for EfficientNet.
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IN_CHANNELS = 6  # 6 cadence observations (ABACAD) stacked depth-wise
    NUM_CLASSES = 1  # Binary classification (Needle vs Haystack)

    # Data Loading
    NUM_WORKERS = 4

    # ==========================================
    # Model Parameters
    # ==========================================
    MODEL_NAME = "efficientnet_b0"
    PRETRAINED = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 32
    NUM_EPOCHS = 12
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2

    # Scheduler (Cosine Annealing)
    T_MAX = NUM_EPOCHS
    ETA_MIN = 1e-6

    # Augmentation
    USE_MIXUP = True
    MIXUP_ALPHA = 1.0

    # ==========================================
    # Hardware
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_directories():
    """
    Creates necessary directories for caching and submissions.
    """
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
