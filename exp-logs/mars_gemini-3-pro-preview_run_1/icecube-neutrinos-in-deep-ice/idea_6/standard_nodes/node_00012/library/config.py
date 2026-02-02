import os
import random
import hashlib
import json
import numpy as np
import torch


class Config:
    """
    Global configuration for the DynGT (Dynamic Graph Transformer) pipeline.
    """

    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"

    # Sub-directories for outputs
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_DIR = os.path.join(WORKING_DIR, "model_checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Input Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.parquet")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.parquet")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.parquet")
    SENSOR_GEO_PATH = os.path.join(INPUT_DIR, "sensor_geometry.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # ==========================================
    # 2. Data Sampling Strategy (Stratified)
    # ==========================================
    # Total number of pulses to sample per event
    N_PULSES = 256

    # Causal Set: First K pulses sorted by time
    K_CAUSAL = 64

    # Signal Set: Top M pulses sorted by charge from the remainder
    M_SIGNAL = 128

    # Context Set: The remaining (N - K - M) pulses are sampled randomly
    # to provide background context.

    # ==========================================
    # 3. Model Architecture (DynGT)
    # ==========================================
    # Input features: [x, y, z, time, log_charge, auxiliary]
    IN_CHANNELS = 6

    # Hidden dimension for transformer and graph layers
    HIDDEN_CHANNELS = 128

    # Number of attention heads
    NUM_HEADS = 8

    # Number of Graph Transformer blocks
    NUM_LAYERS = 6

    # Number of neighbors for dynamic k-NN graph construction
    K_KNN = 20

    # Dropout rate
    DROPOUT = 0.1

    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 192  # Optimized for A100 40GB
    NUM_WORKERS = 4  # Number of dataloader workers
    LEARNING_RATE = 1e-3  # Base learning rate
    WEIGHT_DECAY = 1e-4  # AdamW weight decay
    MAX_EPOCHS = 20  # Maximum training epochs
    PATIENCE = 3  # Early stopping patience
    SEED = 42  # Random seed for reproducibility

    # ==========================================
    # 5. Utilities
    # ==========================================
    @classmethod
    def setup(cls):
        """
        Creates necessary working directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

    @classmethod
    def get_config_hash(cls):
        """
        Generates a unique hash based on data-impacting parameters.
        This is used to version cache files; if sampling logic changes,
        the hash changes, forcing a re-computation of data.
        """
        params = {
            "N_PULSES": cls.N_PULSES,
            "K_CAUSAL": cls.K_CAUSAL,
            "M_SIGNAL": cls.M_SIGNAL,
            "SEED": cls.SEED,
            "IN_CHANNELS": cls.IN_CHANNELS,
        }
        # Sort keys to ensure deterministic hashing
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode("utf-8")).hexdigest()


def set_seed(seed=42):
    """
    Sets fixed random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
