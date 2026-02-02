import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for the Scalar Coupling Prediction Task (Idea 5).
    """

    # ==========================================
    # 1. Environment & Reproducibility
    # ==========================================
    SEED = 42
    NUM_WORKERS = 4  # Number of DataLoader workers

    # ==========================================
    # 2. File Paths & Directories
    # ==========================================
    # Base Directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_5"
    SUBMISSION_DIR = "./submission"

    # Metadata Files (Pre-generated)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
    STRUCTURES_DIR = os.path.join(INPUT_DIR, "structures")
    DIPOLE_MOMENTS_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
    MAGNETIC_SHIELDING_CSV = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
    MULLIKEN_CHARGES_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")
    POTENTIAL_ENERGY_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")
    SCALAR_COUPLING_CONTRIBUTIONS_CSV = os.path.join(
        INPUT_DIR, "scalar_coupling_contributions.csv"
    )
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Processed Data & Cache
    # We use a dedicated directory for cached graph data
    PROCESSED_DATA_DIR = os.path.join(WORKING_DIR, "processed")

    # Specific Cache Files
    CACHE_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_data.pt")
    CACHE_VAL_PATH = os.path.join(PROCESSED_DATA_DIR, "val_data.pt")
    CACHE_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test_data.pt")

    # Statistics for Standardization (Mean/Std per coupling type)
    TARGET_STATS_PATH = os.path.join(WORKING_DIR, "target_stats.npy")

    # Model Artifacts
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 3. Data Processing Constants
    # ==========================================
    COUPLING_TYPES = ["1JHC", "2JHC", "3JHC", "1JHN", "2JHN", "3JHN", "2JHH", "3JHH"]

    # Atom Types Mapping
    ATOM_TYPES = ["H", "C", "N", "O", "F"]
    ATOM_TO_IDX = {atom: i for i, atom in enumerate(ATOM_TYPES)}

    # Graph Construction Parameters
    RADIUS_CUTOFF = 5.0  # Angstroms (standard for QM9/CHAMPS)
    MAX_NEIGHBORS = 32  # Max neighbors for graph connectivity

    # ==========================================
    # 4. Model Hyperparameters
    # ==========================================
    # Radial Basis Functions (Distance)
    RBF_START = 0.0
    RBF_END = 5.0
    NUM_RBF_RADIAL = 128

    # Angular Basis Functions (Cosine of Angle)
    # Cosine values are in [-1, 1]
    NUM_RBF_ANGULAR = 32

    # Network Architecture
    HIDDEN_DIM = 256
    NUM_INTERACTION_LAYERS = 4  # Depth of the Dual-Graph Network
    DROPOUT = 0.1

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 96  # Adjusted for A100 40GB
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-6
    MAX_EPOCHS = 25
    WARMUP_EPOCHS = 2
    PATIENCE = 5  # Early stopping patience

    # Loss Weights
    LAMBDA_PHYS = 0.2  # Weight for auxiliary physics tasks (shielding/charges)

    # Debugging
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 2000

    @classmethod
    def setup_environment(cls):
        """
        Creates necessary directories and sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set Random Seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_device():
        """Returns the PyTorch device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
