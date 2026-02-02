import os
import torch
import random
import numpy as np


class Config:
    # =========================================
    # System & Reproducibility
    # =========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Optimized for 12 vCPUs

    # =========================================
    # File Paths
    # =========================================
    # Input Data (Metadata)
    METADATA_DIR = "./metadata"
    TRAIN_DATA_PATH = os.path.join(METADATA_DIR, "train.parquet")
    VAL_DATA_PATH = os.path.join(METADATA_DIR, "val.parquet")
    TEST_DATA_PATH = os.path.join(METADATA_DIR, "test.parquet")

    # Output / Working Directory
    WORKING_DIR = "./working/idea_5"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.json")
    SCALER_PATH = os.path.join(WORKING_DIR, "scaler_params.npy")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = "./input/sample_submission.csv"

    # =========================================
    # Data Processing & Feature Engineering
    # =========================================
    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100_000  # Number of rows to use if DEBUG is True

    # Spatial Constraints (Strict Bounding Box for NYC)
    # We use a box slightly wider than strict city limits to include airports/suburbs
    # but strict enough to exclude outliers like (0,0).
    # Lat: 39.0 to 42.0, Lon: -75.0 to -72.0
    LAT_MIN = 39.0
    LAT_MAX = 42.0
    LON_MIN = -75.0
    LON_MAX = -72.0
    SPATIAL_BOUNDS = (LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

    # Grid Discretization for Embeddings
    GRID_BINS = 500  # 500x500 grid

    # Target Variable
    TARGET_COL = "fare_amount"
    MIN_FARE_PREDICTION = 2.50  # Floor for predictions

    # =========================================
    # Model Architecture (Deep Residual MLP)
    # =========================================
    EMBEDDING_DIM = 64  # Dimension for spatial embeddings
    HIDDEN_DIM = 512  # Width of residual blocks
    NUM_RES_BLOCKS = 4  # Depth of the network
    DROPOUT_RATE = 0.1

    # =========================================
    # Training Hyperparameters (Robust Optimization)
    # =========================================
    BATCH_SIZE = 4096  # Large batch size for stability
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EPOCHS = 20
    PATIENCE = 3  # Early stopping patience

    # Gradient Handling
    GRAD_CLIP_NORM = 1.0  # Strict clipping to handle outliers

    # Loss Function
    HUBER_DELTA = 1.0  # Transition point for Huber Loss (L2 -> L1)

    @classmethod
    def setup(cls):
        """Creates necessary output directories and sets random seeds."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set reproducible seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_spatial_bounds(cls):
        """Returns spatial bounds as a dictionary for easy access."""
        return {
            "lat_min": cls.LAT_MIN,
            "lat_max": cls.LAT_MAX,
            "lon_min": cls.LON_MIN,
            "lon_max": cls.LON_MAX,
        }
