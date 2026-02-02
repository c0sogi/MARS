import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory (Write Allowed)
    # Specific to Idea 8: Stabilized Dual-Graph Network
    WORK_DIR = "./working/idea_8"

    # Metadata Paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Paths
    STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")
    DIPOLE_MOMENTS_PATH = os.path.join(INPUT_DIR, "dipole_moments.csv")
    MAGNETIC_SHIELDING_PATH = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
    MULLIKEN_CHARGES_PATH = os.path.join(INPUT_DIR, "mulliken_charges.csv")
    POTENTIAL_ENERGY_PATH = os.path.join(INPUT_DIR, "potential_energy.csv")
    CONTRIBUTIONS_PATH = os.path.join(INPUT_DIR, "scalar_coupling_contributions.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(WORK_DIR, "best_model.pth")
    SUBMISSION_PATH = "./submission/submission.csv"

    # Cache Paths for Preprocessed Data
    # We use .npy or .parquet for caching, avoiding pickle
    CACHE_DIR = os.path.join(WORK_DIR, "processed")
    TRAIN_CACHE = os.path.join(
        CACHE_DIR, "train_data.pt"
    )  # PyTorch Geometric data list
    VAL_CACHE = os.path.join(CACHE_DIR, "val_data.pt")
    TEST_CACHE = os.path.join(CACHE_DIR, "test_data.pt")
    STATS_CACHE = os.path.join(CACHE_DIR, "stats.pt")  # Mean/Std for standardization

    # ==========================================
    # 2. Data & Graph Construction
    # ==========================================
    # Graph Construction
    CUTOFF_RADIUS = 5.0  # Angstroms, for Atom Graph
    MAX_NUM_NEIGHBORS = 32

    # Coupling Types
    COUPLING_TYPES = ["1JHC", "2JHH", "1JHN", "2JHN", "2JHC", "3JHC", "3JHH", "3JHN"]

    # Atom Types (H, C, N, O, F) mapped to indices
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # ==========================================
    # 3. Model Architecture (SDG-CFC)
    # ==========================================
    # Dimensions
    HIDDEN_CHANNELS = 128
    NUM_LAYERS = 6  # Depth of the Dual-Graph interaction

    # Continuous Filter Convolutions
    NUM_RBF_DISTANCE = 96  # Basis functions for edge length (Atom Graph)
    NUM_RBF_ANGLE = 32  # Basis functions for cosine angle (Line Graph)

    # Readout
    NUM_HEADS = 8  # One per coupling type

    # ==========================================
    # 4. Training & Optimization
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    BATCH_SIZE = 64  # A100 has 40GB, can handle larger batches
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    MAX_EPOCHS = 40
    PATIENCE = 5  # Early stopping patience

    # Loss Weights
    # Primary loss (coupling) is 1.0 implicitly
    # Auxiliary targets are standardized, so 0.1 is a balanced weight
    AUX_LOSS_WEIGHT = 0.1

    # Scheduler
    MIN_LR = 1e-6

    # Debugging
    DEBUG = False
    DEBUG_SAMPLES = 2000  # Number of samples to use if DEBUG is True

    @classmethod
    def setup(cls):
        """
        Creates necessary directories and sets random seeds.
        """
        # Create directories
        os.makedirs(cls.WORK_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_coupling_type_id(cls, type_str):
        """Maps coupling type string to integer index."""
        try:
            return cls.COUPLING_TYPES.index(type_str)
        except ValueError:
            raise ValueError(f"Unknown coupling type: {type_str}")
