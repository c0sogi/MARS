import os
import torch
import random
import numpy as np


class Config:
    """
    Configuration class for the Hybrid Geometric-Attention Network (HGA-Net) pipeline.
    Centralizes all hyperparameters, file paths, and experiment settings.
    """

    # =========================================================================
    # General Settings
    # =========================================================================
    EXPERIMENT_NAME = "idea_8"
    SEED = 42
    DEBUG = False  # Set to True for fast debugging with a small data subset
    DEBUG_SUBSET_SIZE = 2000  # Number of molecules to use when DEBUG is True

    # Compute
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    # =========================================================================
    # File System Paths
    # =========================================================================
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    STRUCTURES_DIR = os.path.join(INPUT_DIR, "structures")

    # Metadata Files (Pre-generated splits)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Files
    STRUCTURES_CSV_PATH = os.path.join(INPUT_DIR, "structures.csv")

    # Output Directories
    WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
    SUBMISSION_DIR = "./submission"

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Artifact Paths
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
    SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Caching Paths (for deterministic data processing)
    # Using .npz for efficient storage of processed graph data
    CACHE_TRAIN_PATH = os.path.join(WORKING_DIR, "cached_train.npz")
    CACHE_VAL_PATH = os.path.join(WORKING_DIR, "cached_val.npz")
    CACHE_TEST_PATH = os.path.join(WORKING_DIR, "cached_test.npz")
    CACHE_STATS_PATH = os.path.join(WORKING_DIR, "target_stats.npz")

    # =========================================================================
    # Data Processing Hyperparameters
    # =========================================================================
    # Graph Construction
    SPATIAL_CUTOFF = 5.0  # Angstroms (radius for connecting atoms)
    MAX_NEIGHBORS = 32  # Maximum neighbors per node to maintain efficiency

    # Geometric Basis Functions
    NUM_RBF = 128  # Number of Radial Basis Functions for distance
    RBF_START = 0.0
    RBF_END = 5.0
    NUM_SBF = 7  # Spherical Harmonics L_max for angle representation

    # Normalization
    # Targets are standardized per coupling type (mean=0, std=1)

    # =========================================================================
    # Model Architecture: HGA-Net
    # =========================================================================
    # Backbone: Directional Message Passing (DMPNN)
    HIDDEN_DIM = 256  # Dimension of node and edge embeddings
    NUM_MP_LAYERS = 6  # Number of message passing interaction layers

    # Global Interaction: Transformer
    TRANSFORMER_LAYERS = 3
    TRANSFORMER_HEADS = 8
    TRANSFORMER_FF_DIM = 512

    # Readout
    DROPOUT = 0.0  # No dropout to preserve geometric precision

    # =========================================================================
    # Training Hyperparameters
    # =========================================================================
    BATCH_SIZE = 32
    EPOCHS = 50  # Train for 50-60 epochs as per strategy
    LEARNING_RATE = 1e-4  # Peak learning rate
    WEIGHT_DECAY = 1e-5  # L2 Regularization
    WARMUP_EPOCHS = 3  # Linear warmup duration
    GRAD_CLIP = 1.0  # Gradient clipping threshold

    # =========================================================================
    # Helper Methods
    # =========================================================================
    @staticmethod
    def set_seed(seed=42):
        """Sets fixed random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
