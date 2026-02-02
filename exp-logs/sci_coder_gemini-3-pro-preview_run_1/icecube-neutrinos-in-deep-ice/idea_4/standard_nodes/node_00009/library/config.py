import os
import json
import hashlib
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Neutrino Direction Prediction pipeline.
    Centralizes all parameters for data processing, model architecture, and training.
    """

    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_4"
    SUBMISSION_DIR = "./submission"

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")

    # Geometry Path
    SENSOR_GEO_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")

    # Output Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data Processing Hyperparameters
    # ==========================================
    SEED = 42

    # Sampling: How many pulses to keep per event to form the graph
    MAX_PULSES = 192

    # Graph Construction: Number of nearest neighbors for EdgeConv
    K_NEIGHBORS = 10

    # Filtering: Whether to exclude auxiliary (low quality) pulses
    FILTER_AUXILIARY = True

    # Normalization Constants (Derived from data analysis)
    # Time is relative (0 to ~30k ns), Position is in meters (-500 to 500)
    NORM_TIME_SCALE = 30000.0
    NORM_POS_SCALE = 600.0

    # ==========================================
    # 3. Model Architecture
    # ==========================================
    # Input features: [x, y, z, time, log_charge]
    INPUT_DIM = 5

    # Hidden dimension for GNN layers
    HIDDEN_DIM = 128

    # Number of Dynamic EdgeConv layers
    NUM_LAYERS = 4

    # Dropout rate
    DROPOUT = 0.1

    # Output dimension: 3D vector (x, y, z)
    OUTPUT_DIM = 3

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 20
    PATIENCE = 4  # Early stopping patience

    # Compute resources
    NUM_WORKERS = 12  # Utilizing available vCPUs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 5. Utilities
    # ==========================================
    @classmethod
    def get_config_hash(cls):
        """
        Generates a unique MD5 hash based on the data processing configuration.
        This hash is used to create unique filenames for cached datasets.
        If any data parameter changes, the hash changes, invalidating old caches.
        """
        config_dict = {
            "max_pulses": cls.MAX_PULSES,
            "k_neighbors": cls.K_NEIGHBORS,
            "filter_aux": cls.FILTER_AUXILIARY,
            "seed": cls.SEED,
            "norm_time": cls.NORM_TIME_SCALE,
            "norm_pos": cls.NORM_POS_SCALE,
        }
        # Sort keys to ensure deterministic hashing
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    @classmethod
    def initialize(cls):
        """
        Performs initial setup:
        1. Creates necessary working directories.
        2. Sets fixed random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Automatically initialize environment on import
Config.initialize()
