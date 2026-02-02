import os
import sys
import random
import logging
import numpy as np
import torch


class Config:
    """
    Configuration class containing all hyperparameters and constants for the
    Hierarchical ConvNeXt-Base Strategy.
    """

    # -------------------------------------------------------------------------
    # Directories & Files
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    OUTPUT_DIR = "./submission"

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    TRAIN_META_JSON = os.path.join(INPUT_DIR, "train_metadata.json")

    # -------------------------------------------------------------------------
    # Dataset & Taxonomy
    # -------------------------------------------------------------------------
    IMAGE_SIZE = 320
    NUM_CLASSES_SPECIES = 15501
    NUM_CLASSES_GENUS = 2564
    NUM_CLASSES_FAMILY = 272

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # ConvNeXt Base initialized with ImageNet-21k weights
    MODEL_NAME = "convnext_base.fb_in22k"

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 32  # Physical batch size per GPU (A100 40GB)
    ACCUMULATION_STEPS = 2  # Effective Batch Size = 64
    EPOCHS = 12  # Total training epochs
    NUM_WORKERS = 12  # Available vCPUs

    # Optimization
    LR_BACKBONE = 1e-5  # Lower LR for pre-trained backbone
    LR_HEAD = 1e-3  # Higher LR for new classification heads
    WEIGHT_DECAY = 1e-4

    # Loss & Regularization
    LABEL_SMOOTHING = 0.1
    LAMBDA_SPECIES = 1.0  # Main task weight
    LAMBDA_GENUS = 0.5  # Auxiliary task weight
    LAMBDA_FAMILY = 0.5  # Auxiliary task weight

    # Stochastic Weight Averaging (SWA)
    USE_SWA = True
    SWA_START_EPOCH = 8  # Start SWA at this epoch
    SWA_LR = 1e-5  # Learning rate during SWA phase

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 5000  # Number of samples to use when DEBUG is True


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic algorithms in cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """
    Returns the appropriate PyTorch device (CUDA if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_logger(name, log_file, level=logging.INFO):
    """
    Configures and returns a logger that writes to both the console and a file.

    Args:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        level (int): Logging level.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(level)
    c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # File Handler
    f_handler = logging.FileHandler(log_file)
    f_handler.setLevel(level)
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger


def ensure_dirs():
    """
    Creates the necessary working and output directories specified in Config.
    """
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
