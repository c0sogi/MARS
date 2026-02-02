import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # 1. Paths & Directories
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"
    SUBMISSION_DIR = "./submission"

    # Input Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
    DIPOLE_MOMENTS_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
    MAGNETIC_SHIELDING_CSV = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
    MULLIKEN_CHARGES_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")
    POTENTIAL_ENERGY_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Caching Paths (using .npy and .parquet as requested)
    # We will store processed graph data as separate numpy arrays
    CACHE_NODES_PATH = os.path.join(WORKING_DIR, "cached_nodes.npy")
    CACHE_EDGES_PATH = os.path.join(WORKING_DIR, "cached_edges.npy")
    CACHE_EDGE_ATTR_PATH = os.path.join(WORKING_DIR, "cached_edge_attrs.npy")
    CACHE_TRIPLETS_PATH = os.path.join(WORKING_DIR, "cached_triplets.npy")  # For angles
    CACHE_GRAPH_INDICES_PATH = os.path.join(
        WORKING_DIR, "cached_graph_indices.npy"
    )  # To reconstruct batch
    CACHE_TARGETS_PATH = os.path.join(WORKING_DIR, "cached_targets.npy")
    CACHE_AUX_TARGETS_PATH = os.path.join(
        WORKING_DIR, "cached_aux_targets.npy"
    )  # For shielding/charges
    CACHE_META_DF_PATH = os.path.join(WORKING_DIR, "cached_meta.parquet")

    # Statistics for Normalization
    STATS_PATH = os.path.join(WORKING_DIR, "target_stats.npy")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Model Hyperparameters
    # ==========================================
    # Architecture
    HIDDEN_DIM = 128
    NUM_LAYERS = 4  # Depth of message passing
    NUM_RBF = 32  # Number of Radial Basis Functions
    NUM_SBF = 7  # Number of Spherical Bessel Functions
    CUTOFF = 5.0  # Interaction radius in Angstroms

    # Readout
    NUM_COUPLING_TYPES = 8  # 1JHC, 2JHC, etc.

    # ==========================================
    # 3. Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 32  # Graph batch size
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-5
    MAX_EPOCHS = 30
    PATIENCE = 5  # Early stopping patience

    # Auxiliary Loss Weights
    # L_total = L_coupling + lambda1 * L_shielding + lambda2 * L_charge
    LAMBDA_SHIELDING = 0.15
    LAMBDA_CHARGE = 0.10

    # ==========================================
    # 4. Physics & Data Constants
    # ==========================================
    # Atom Type Mapping (Atomic Number)
    ATOM_MAP = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
    MAX_ATOMIC_NUM = 9  # For embedding layer size

    # Coupling Type Mapping
    COUPLING_TYPES = ["1JHC", "2JHC", "3JHC", "1JHN", "2JHN", "3JHN", "2JHH", "3JHH"]
    COUPLING_TYPE_MAP = {t: i for i, t in enumerate(COUPLING_TYPES)}

    def __init__(self):
        # Ensure directories exist
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def set_seed(seed=42):
        """Sets the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def get_device():
        """Returns the appropriate device."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
