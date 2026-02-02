import os
import torch
import random
import numpy as np
import math


# --------------------------------------------------------------------------
# Patch for NumPy 2.0 Compatibility
# --------------------------------------------------------------------------
# NumPy 2.0 removed np.math (alias for python math module).
# Some libraries (e.g., torch_geometric < 2.7) still rely on it.
if not hasattr(np, "math"):
    np.math = math


class Config:
    # --------------------------------------------------------------------------
    # General Settings
    # --------------------------------------------------------------------------
    PROJECT_NAME = "idea_7"
    SEED = 42
    DEBUG = False  # Set to True to run on a small subset for testing
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # For data loading

    # --------------------------------------------------------------------------
    # File Paths
    # --------------------------------------------------------------------------
    # Input Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    STRUCTURES_DIR = os.path.join(INPUT_DIR, "structures")

    # Metadata Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Auxiliary Data
    DIPOLE_MOMENTS_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
    POTENTIAL_ENERGY_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")
    MULLIKEN_CHARGES_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")

    # Output Directories
    WORKING_DIR = f"./working/{PROJECT_NAME}"
    CACHE_DIR = WORKING_DIR  # For caching processed graphs
    SUBMISSION_DIR = "./submission"

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters (HGA-Net)
    # --------------------------------------------------------------------------
    # Backbone (Directional Message Passing)
    HIDDEN_DIM = 256  # High capacity as per strategy
    NUM_MP_LAYERS = 6  # Depth of message passing
    CUTOFF = 5.0  # Spatial cutoff in Angstroms
    NUM_RBF = 64  # Number of Radial Basis Functions
    NUM_SBF = 7  # Number of Spherical Basis Functions

    # Global Interaction (Transformer)
    TRANSFORMER_LAYERS = 3  # Number of self-attention layers
    TRANSFORMER_HEADS = 8  # Number of attention heads
    TRANSFORMER_DIM_FEEDFORWARD = 512

    # Readout
    DROPOUT = 0.0  # Deterministic MLP (No dropout)

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 128  # Adjusted for A100 memory with large graphs
    LEARNING_RATE = 5e-4  # Initial learning rate
    WEIGHT_DECAY = 1e-5  # Regularization
    MAX_EPOCHS = 50  # Extended duration for convergence
    WARMUP_EPOCHS = 3  # Linear warmup
    PATIENCE = 10  # Early stopping patience

    # --------------------------------------------------------------------------
    # Data Processing
    # --------------------------------------------------------------------------
    # If Debug is True, only use this many molecules
    DEBUG_SAMPLE_SIZE = 1000

    # Normalization constants (will be computed or loaded)
    # Dictionary mapping coupling type to (mean, std)
    TARGET_NORMALIZATION = {}

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)


# Initialize seed immediately
Config.set_seed(Config.SEED)
