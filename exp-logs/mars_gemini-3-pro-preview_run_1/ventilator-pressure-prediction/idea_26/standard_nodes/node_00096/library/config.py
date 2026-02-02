import os
import torch
import random
import numpy as np


class Config:
    """
    Global configuration for the Ventilator Pressure Prediction task.
    Implements the 'Wide-Projected Deeply-Supervised Physics-Identity Network' strategy.
    """

    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_27"
    SUBMISSION_DIR = "./submission"

    # Raw Data Paths (using metadata splits)
    TRAIN_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_PATH = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Paths (Parquet for Dataframes, NPY for Scaler stats)
    # We avoid pickle as per requirements.
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "dataset_train_engineered.parquet")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "dataset_val_engineered.parquet")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "dataset_test_engineered.parquet")

    # Scaler statistics (center and scale) stored as numpy arrays
    SCALER_CENTER_PATH = os.path.join(WORKING_DIR, "scaler_center.npy")
    SCALER_SCALE_PATH = os.path.join(WORKING_DIR, "scaler_scale.npy")

    # =========================================================================
    # Model Architecture Hyperparameters
    # =========================================================================
    # Strategy: Wide-Projected Deeply-Supervised Physics-Identity Network

    # Stem / Input
    STEM_DIM = 512  # Dimension after initial convolution and projection

    # Backbone
    MODEL_DIM = 1024  # High-capacity latent dimension
    HIDDEN_DIM = 512  # Bi-LSTM hidden size (512 * 2 directions = 1024 output)
    NUM_BLOCKS = 4  # Number of composite blocks
    EXPANSION_FACTOR = 2  # FFN expansion (2 * 1024 = 2048)
    DROPOUT = 0.1

    # Auxiliary Supervision
    AUX_BLOCK_INDEX = 2  # Attach aux head after the 3rd block (index 2)
    AUX_LOSS_WEIGHT = 0.3  # Weight for the auxiliary loss

    # =========================================================================
    # Feature Engineering
    # =========================================================================
    USE_LAGS = True
    LAG_STEPS = 4  # Lags 1, 2, 3, 4
    USE_DIFFS = True  # First and Second differences of u_in
    USE_PHYSICS = True  # R*u_in, Volume/C, etc.

    # Dimensions
    SEQ_LEN = 80  # Fixed sequence length per breath

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    SEED = 42
    EPOCHS = 35  # Extended horizon for OneCycleLR
    BATCH_SIZE = 512  # Update budget
    LEARNING_RATE = 1e-3  # Max LR for OneCycle
    WEIGHT_DECAY = 1e-2
    GRAD_CLIP = 1.0  # Strict clipping for stability
    PATIENCE = 7  # Early stopping patience
    NUM_WORKERS = 4  # Data loading workers

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================================================================
    # Debugging / Development
    # =========================================================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLES = 2000  # Number of breaths to use in debug mode

    @classmethod
    def setup(cls):
        """
        Initialize the environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(
            f"Config setup complete. Device: {cls.DEVICE}, Working Dir: {cls.WORKING_DIR}"
        )

    @classmethod
    def get_scaler_paths(cls):
        """Returns tuple of paths for scaler center and scale arrays."""
        return cls.SCALER_CENTER_PATH, cls.SCALER_SCALE_PATH
