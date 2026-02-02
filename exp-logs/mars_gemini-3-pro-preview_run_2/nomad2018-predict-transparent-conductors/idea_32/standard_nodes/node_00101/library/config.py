import os
import torch


class Config:
    """
    Configuration class for the Multi-Scale Receiver-Aware Crystal Graph Network (MS-RA-CGN).
    Centralizes all hyperparameters for data processing, model architecture, and training.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific experiment
    WORKING_DIR = "./working/idea_32"

    # Metadata file paths (pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"  # Root submission directory as per requirements

    # Ensure necessary directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Radius Graph Construction
    CUTOFF = 5.0  # Angstroms (Strict cutoff for local constraints)
    MAX_NEIGHBORS = 50  # Limit neighbors for memory efficiency

    # Atom Features
    MAX_ATOMIC_NUMBER = 100  # Size of atomic number embedding lookup table

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters (MS-RA-CGN)
    # -------------------------------------------------------------------------
    # General Dimensions
    HIDDEN_DIM = 128  # Node and Edge embedding dimension

    # Multi-Scale RBF Encoder Settings
    # Fine-scale bank for precise bond lengths
    RBF_FINE_BINS = 64
    RBF_FINE_SIGMA = 0.08

    # Coarse-scale bank for general proximity
    RBF_COARSE_BINS = 32
    RBF_COARSE_SIGMA = 0.5

    # RBF Range (matches cutoff)
    RBF_START = 0.0
    RBF_END = 5.0

    # Network Depth
    NUM_LAYERS = 4  # Number of Interaction Blocks

    # Regularization
    DROPOUT = 0.1  # Applied in interaction blocks and readout heads

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 48  # Selected for gradient noise balance
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Decoupled weight decay

    # Training Loop Control
    MAX_EPOCHS = 150
    PATIENCE = 20  # Early stopping patience

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 8
    MIN_LR = 1e-6

    # Targets
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # -------------------------------------------------------------------------
    # Debugging / Development / Hardware
    # -------------------------------------------------------------------------
    # Set to a small integer (e.g., 100) to run on a subset of data for debugging
    # Set to None to run on full dataset
    DEBUG_SAMPLE_SIZE = None

    # Compute configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 50)
        print("MS-RA-CGN Configuration")
        print("=" * 50)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<25}: {v}")
        print("=" * 50)
