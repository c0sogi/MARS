import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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
    Global configuration for the Neutrino Direction Prediction pipeline.
    """

    # ==========================================
    # Directories and Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    IDEA_DIR = os.path.join(WORKING_DIR, "idea_1")
    SUBMISSION_DIR = "./submission"

    # Create working directories if they don't exist
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")

    # Geometry and Submission Files
    GEOMETRY_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Hyperparameters
    # ==========================================
    # Sequence length for the 1D CNN input
    SEQ_LEN = 128

    # Input features: x, y, z, time, charge (auxiliary is filtered out)
    NUM_FEATURES = 5

    # Normalization Constants (Derived from Data Analysis)
    # Time: Mean ~12900, Std ~4400
    NORM_TIME_MEAN = 12900.0
    NORM_TIME_STD = 4400.0

    # Coordinates: IceCube sensors are roughly centered at (0,0,0) with a spread of ~500m
    NORM_COORD_MEAN = 0.0
    NORM_COORD_STD = 500.0

    # Charge: We will use log1p(charge) in the dataset, so explicit scaling constants
    # for raw charge are not strictly necessary here, but can be added if needed.

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    INPUT_DIM = 5  # (x, y, z, time, charge)
    OUTPUT_DIM = 3  # Predicting 3D direction vector (x, y, z)

    HIDDEN_DIM = 128
    NUM_LAYERS = 4
    KERNEL_SIZE = 3
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5

    # Debugging / Subsetting
    # Set DEBUG to True to train on a small subset of data for quick iteration
    DEBUG = False
    DEBUG_SUBSET_SIZE = 50000

    # ==========================================
    # Hardware / Compute
    # ==========================================
    SEED = 42
    # Reduced from 8 to 4 to prevent OOM errors (ArrowMemoryError)
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
