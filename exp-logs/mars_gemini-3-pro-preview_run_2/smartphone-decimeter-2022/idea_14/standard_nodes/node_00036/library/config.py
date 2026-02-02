import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Sky-State Transformer for Anchor-Free Trajectory Refinement.
    """

    # -------------------------------------------------------------------------
    # Directory Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    WINDOW_SIZE = 15  # Size of the sliding window (N)

    # Features to be used in the Trajectory Stream (Sequence)
    # These will be computed per timestep in the window
    SEQ_FEATURES = [
        "rel_lat_m",  # Relative Latitude in meters (centered at window mid)
        "rel_lon_m",  # Relative Longitude in meters (centered at window mid)
        "rel_alt_m",  # Relative Altitude in meters (centered at window mid)
        "delta_lat_m",  # Velocity North (m/s)
        "delta_lon_m",  # Velocity East (m/s)
        "delta_alt_m",  # Velocity Up (m/s)
        "Cn0DbHz",  # Signal Strength
        "RawPseudorangeUncertaintyMeters",  # Uncertainty
    ]

    # Features to be used in the Sky-State Context Stream (Aggregated)
    # These will be aggregated over the window (Mean and Std)
    SKY_FEATURES = [
        "mean_Cn0",
        "std_Cn0",
        "mean_SvElevation",
        "std_SvElevation",
        "mean_SvAzimuth",
        "std_SvAzimuth",
        "mean_Uncertainty",
        "std_Uncertainty",
        "mean_SignalCount",
    ]

    # -------------------------------------------------------------------------
    # Caching Paths (for deterministic data processing)
    # -------------------------------------------------------------------------
    # Training Data Cache
    CACHE_TRAIN_X_SEQ = os.path.join(WORKING_DIR, "train_X_seq.npy")
    CACHE_TRAIN_X_SKY = os.path.join(WORKING_DIR, "train_X_sky.npy")
    CACHE_TRAIN_Y = os.path.join(WORKING_DIR, "train_y.npy")
    CACHE_TRAIN_META = os.path.join(WORKING_DIR, "train_meta.parquet")

    # Validation Data Cache
    CACHE_VAL_X_SEQ = os.path.join(WORKING_DIR, "val_X_seq.npy")
    CACHE_VAL_X_SKY = os.path.join(WORKING_DIR, "val_X_sky.npy")
    CACHE_VAL_Y = os.path.join(WORKING_DIR, "val_y.npy")
    CACHE_VAL_META = os.path.join(WORKING_DIR, "val_meta.parquet")

    # Test Data Cache
    CACHE_TEST_X_SEQ = os.path.join(WORKING_DIR, "test_X_seq.npy")
    CACHE_TEST_X_SKY = os.path.join(WORKING_DIR, "test_X_sky.npy")
    CACHE_TEST_META = os.path.join(WORKING_DIR, "test_meta.parquet")

    # Scaler Cache
    CACHE_SCALER = os.path.join(WORKING_DIR, "scaler.json")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Transformer Backbone
    TRANSFORMER_HIDDEN_SIZE = 128
    TRANSFORMER_NUM_HEADS = 4
    TRANSFORMER_NUM_LAYERS = 2
    TRANSFORMER_DROPOUT = 0.1

    # Fusion & Output Head
    SKY_EMBED_SIZE = 32
    MLP_HIDDEN_SIZE = 64
    OUTPUT_DIM = 2  # Delta East, Delta North (Meters)

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 10
    RANDOM_STATE = 42

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    # Debugging / Development
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 1000  # Number of samples to use if DEBUG is True

    @staticmethod
    def set_seed():
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(Config.RANDOM_STATE)
        np.random.seed(Config.RANDOM_STATE)
        torch.manual_seed(Config.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.RANDOM_STATE)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
