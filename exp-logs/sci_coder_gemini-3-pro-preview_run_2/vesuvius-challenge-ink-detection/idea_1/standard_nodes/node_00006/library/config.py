import os
import torch
import random
import numpy as np


class Config:
    # ==============================
    # Directory & File Paths
    # ==============================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and model checkpoints
    WORKING_DIR = "./working/idea_2"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VALID_METADATA_PATH = os.path.join(METADATA_DIR, "validation.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output submission path
    SUBMISSION_PATH = "./submission/submission.csv"

    # ==============================
    # Data Parameters
    # ==============================
    TILE_SIZE = 512
    STRIDE = 512  # Stride for tiling (non-overlapping)

    # Z-slice range for Maximum Intensity Projection (MIP)
    # The ink is typically found in the central layers of the volume
    Z_START = 22
    Z_END = 42

    # ==============================
    # Model Parameters
    # ==============================
    ENCODER_NAME = "resnet18"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 1  # Grayscale input (MIP)
    CLASSES = 1  # Binary segmentation

    # ==============================
    # Training Hyperparameters
    # ==============================
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    SEED = 42

    # Early Stopping parameters
    PATIENCE = 5

    # ==============================
    # Compute
    # ==============================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use available CPUs for data loading
    NUM_WORKERS = 2


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
