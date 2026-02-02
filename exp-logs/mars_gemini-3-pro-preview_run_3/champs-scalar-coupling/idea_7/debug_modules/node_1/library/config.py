import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 12 vCPUs available, leaving some overhead
    NUM_WORKERS = 8

    # ==========================================
    # File Paths
    # ==========================================
    # Read-only Input Directories
    INPUT_DIR = "./input"
    STRUCTURES_DIR = os.path.join(INPUT_DIR, "structures")
    METADATA_DIR = "./metadata"

    # Read-only Input Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
    DIPOLE_MOMENTS_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
    MAGNETIC_SHIELDING_CSV = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
    MULLIKEN_CHARGES_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")
    POTENTIAL_ENERGY_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")

    # Metadata Splits (Generated previously)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Writable Working Directory (Cache & Outputs)
    WORKING_DIR = "./working/idea_7"
    PROCESSED_DIR = os.path.join(WORKING_DIR, "processed")
    SUBMISSION_DIR = "./submission"

    # Ensure writable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Graph Construction
    ATOM_GRAPH_CUTOFF = 5.0  # Angstroms for neighbor search
    MAX_MOL_ATOMS = 32  # Max atoms per molecule (QM9 max is ~29)

    # Continuous Filter Convolution (CFConv) Basis
    RBF_MIN_DIST = 0.0
    RBF_MAX_DIST = 5.0
    NUM_RBF_DIST = 64  # Number of Gaussian basis functions for distance
    NUM_RBF_ANGLE = 32  # Number of Gaussian basis functions for angles

    # Coupling Types
    COUPLING_TYPES = ["1JHC", "2JHC", "3JHC", "1JHN", "2JHN", "3JHN", "2JHH", "3JHH"]
    # Mapping for routing to specific heads
    COUPLING_TYPE_MAP = {t: i for i, t in enumerate(COUPLING_TYPES)}

    # ==========================================
    # Model Architecture (S-GLGN)
    # ==========================================
    HIDDEN_DIM = 128
    NUM_LAYERS = 4
    NUM_HEADS = 8  # One head per coupling type
    DROPOUT = 0.0  # Dropout rate

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 128  # Optimized for A100-40GB
    LEARNING_RATE = 5e-4  # Initial LR for AdamW
    WEIGHT_DECAY = 1e-6
    EPOCHS = 25
    PATIENCE = 5  # Early stopping patience
    AUX_LOSS_WEIGHT = 0.1  # Weight lambda for auxiliary tasks (Shielding/Charges)

    # ==========================================
    # Debugging & Development
    # ==========================================
    DEBUG = False  # Set to True to use a small subset for testing
    DEBUG_SAMPLE_SIZE = 5000  # Number of samples to use if DEBUG is True

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
