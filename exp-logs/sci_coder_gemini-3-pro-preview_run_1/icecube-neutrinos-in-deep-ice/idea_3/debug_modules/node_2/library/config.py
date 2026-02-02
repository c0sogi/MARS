import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")

    # Geometry File
    SENSOR_GEOMETRY_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")

    # Output Files
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Caching
    CACHE_DIR = WORKING_DIR

    # ==========================================
    # Data Configuration
    # ==========================================
    # Sequence Processing
    SEQ_LEN = 196

    # Features
    # Sequence Branch: x, y, z, time, charge
    SEQ_FEATURES = ["x", "y", "z", "time", "charge"]
    N_SEQ_FEATURES = len(SEQ_FEATURES)

    # Wide Branch (Engineered):
    # charge-weighted centroids (x, y, z), time quantiles (10th, 50th), total charge
    MANUAL_FEATURES = [
        "center_x",
        "center_y",
        "center_z",
        "time_q10",
        "time_q50",
        "total_charge",
    ]
    N_MANUAL_FEATURES = len(MANUAL_FEATURES)

    # Debugging / Sampling
    DEBUG = False  # Set to True to run on a small subset
    MAX_SAMPLES = 50000 if DEBUG else None  # None means use full dataset

    # ==========================================
    # Model Architecture
    # ==========================================
    # GRU Branch
    GRU_HIDDEN_DIM = 256
    GRU_NUM_LAYERS = 2
    GRU_BIDIRECTIONAL = True
    GRU_DROPOUT = 0.1

    # MLP Branch
    MLP_HIDDEN_DIM = 128

    # Fusion
    FUSION_HIDDEN_DIM = 128
    DROPOUT_RATE = 0.2

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 10
    NUM_WORKERS = 12  # Utilizing all 12 vCPUs

    # Scheduler (OneCycleLR)
    PCT_START = 0.3
    DIV_FACTOR = 25.0
    FINAL_DIV_FACTOR = 1000.0

    # Early Stopping
    PATIENCE = 3

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across numpy, torch, and python.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
