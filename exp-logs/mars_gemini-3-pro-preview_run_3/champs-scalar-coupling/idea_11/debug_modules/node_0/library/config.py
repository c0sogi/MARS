import os
import torch


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_11"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Input Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
    DIPOLE_MOMENTS_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
    MAGNETIC_SHIELDING_CSV = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
    MULLIKEN_CHARGES_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")
    POTENTIAL_ENERGY_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Generated previously)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Paths (for SoA flattened arrays)
    # We use a subdirectory in working to keep things organized
    PROCESSED_DATA_DIR = os.path.join(WORKING_DIR, "processed_soa")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Specific Cache Files
    CACHE_NODES_PATH = os.path.join(PROCESSED_DATA_DIR, "nodes.npy")
    CACHE_EDGES_PATH = os.path.join(PROCESSED_DATA_DIR, "edge_indices.npy")
    CACHE_EDGE_ATTRS_PATH = os.path.join(
        PROCESSED_DATA_DIR, "edge_attrs.npy"
    )  # Distances/vectors
    CACHE_TRIPLETS_PATH = os.path.join(PROCESSED_DATA_DIR, "triplets.npy")  # For angles
    CACHE_MOL_INDICES_PATH = os.path.join(PROCESSED_DATA_DIR, "mol_indices.npy")

    # Target Cache Files
    CACHE_TRAIN_TARGETS_PATH = os.path.join(PROCESSED_DATA_DIR, "train_targets.npy")
    CACHE_VAL_TARGETS_PATH = os.path.join(PROCESSED_DATA_DIR, "val_targets.npy")
    CACHE_AUX_TARGETS_PATH = os.path.join(
        PROCESSED_DATA_DIR, "aux_targets.npy"
    )  # Shielding/Charges

    # Stats Cache
    STATS_PATH = os.path.join(PROCESSED_DATA_DIR, "stats.npy")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Reproducibility
    # ==========================================
    SEED = 42

    # ==========================================
    # 3. Data Definitions & Mappings
    # ==========================================
    # Atom mapping (based on QM9/CHamps standard elements)
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # Coupling Type mapping (from EDA)
    COUPLING_TYPES = ["1JHC", "2JHH", "1JHN", "2JHN", "2JHC", "3JHH", "3JHC", "3JHN"]
    TYPE_MAP = {t: i for i, t in enumerate(COUPLING_TYPES)}
    NUM_COUPLING_TYPES = len(COUPLING_TYPES)

    # ==========================================
    # 4. Model Hyperparameters
    # ==========================================
    # Geometric Graph Settings
    RADIUS_CUTOFF = 5.0  # Angstroms, critical for capturing relevant environment
    MAX_NEIGHBORS = 32  # Cap on neighbors for efficiency

    # Architecture Dimensions
    HIDDEN_DIM = 256
    NUM_LAYERS = 4  # Depth of Message Passing
    NUM_RBF = 64  # Number of Radial Basis Functions for distance expansion
    NUM_ABF = 32  # Number of Angular Basis Functions (if used) or RBF for angles

    # Readout
    DROPOUT = 0.1

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # Optimization
    BATCH_SIZE = 192  # Fits in A100 comfortably with flattened arrays
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5

    # Scheduler (Cosine Annealing Warm Restarts)
    T_0 = 10  # First restart period
    T_MULT = 2  # Multiplier for period
    ETA_MIN = 1e-6  # Min LR

    # Training Loop
    MAX_EPOCHS = 25  # Sufficient for convergence given 24h limit
    EARLY_STOPPING_PATIENCE = 5

    # Loss Weights
    # Total = L1_coupling + LAMBDA * (L1_shielding + L1_charge)
    AUX_LOSS_WEIGHT = 0.1

    # ==========================================
    # 6. Debugging & Development
    # ==========================================
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 5000  # Number of molecules to use if DEBUG is True

    def __init__(self):
        # Set seeds immediately upon instantiation
        self.set_seed()

    @classmethod
    def set_seed(cls):
        import numpy as np
        import random

        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def print_config(cls):
        print("\n=== Configuration ===")
        print(f"Device: {cls.DEVICE}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Hidden Dim: {cls.HIDDEN_DIM}")
        print(f"Radius Cutoff: {cls.RADIUS_CUTOFF}")
        print(f"Working Dir: {cls.WORKING_DIR}")
        print(f"Debug Mode: {cls.DEBUG}")
        print("=====================\n")
