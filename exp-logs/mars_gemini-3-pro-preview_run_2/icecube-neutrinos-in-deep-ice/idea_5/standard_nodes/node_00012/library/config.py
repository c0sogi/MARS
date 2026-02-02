import os
import torch
import random
import numpy as np


class Config:
    """
    Central configuration for the Dual-Stream Geometric-Temporal Network.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SENSOR_GEO_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # =========================================================================
    # Data Parameters
    # =========================================================================
    # Stream 1: Sequence Data
    SEQ_LEN = 192  # Top N pulses by charge
    N_CHANNELS = 6  # [x, y, z, time, charge, auxiliary]

    # Stream 2: Geometric Features
    # 3 (Center of Gravity) + 6 (Covariance Matrix: xx, yy, zz, xy, xz, yz)
    NUM_GEOM_FEATURES = 9

    # =========================================================================
    # Model Hyperparameters
    # =========================================================================
    RESNET_BASE_FILTERS = 64
    MLP_HIDDEN_DIM = 256
    DROPOUT_RATE = 0.1

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 256  # A100 40GB allows for large batches
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 4
    WARMUP_EPOCHS = 2
    WEIGHT_DECAY = 1e-4

    # =========================================================================
    # Compute & Reproducibility
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12
    SEED = 42

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    # Set to True to run on a small subset of data for pipeline verification
    DEBUG = False
    DEBUG_SUBSET_SIZE = 10000

    @classmethod
    def setup(cls):
        """Ensures necessary working directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


def seed_everything(seed: int = 42):
    """
    Sets seeds for all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
