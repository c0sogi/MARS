import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for Scalar Coupling Prediction.
    Implements the settings for the Scalable Directional Message Passing strategy.
    """

    # ==========================================
    # Global Settings
    # ==========================================
    SEED = 42
    DEBUG = False  # Set True to run on a small subset for debugging
    DEBUG_SAMPLE_SIZE = 5000  # Number of molecules for debugging

    # ==========================================
    # Directory Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for caching processed data and models
    # Using 'idea_13' as per the caching requirements in the prompt
    WORKING_DIR = "./working/idea_13"

    # Sub-directories
    PROCESSED_DIR = os.path.join(WORKING_DIR, "processed_soa")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure directories exist
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # ==========================================
    # File Paths
    # ==========================================
    # Raw Input Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
    DIPOLE_MOMENTS_CSV = os.path.join(INPUT_DIR, "dipole_moments.csv")
    MAGNETIC_SHIELDING_CSV = os.path.join(INPUT_DIR, "magnetic_shielding_tensors.csv")
    MULLIKEN_CHARGES_CSV = os.path.join(INPUT_DIR, "mulliken_charges.csv")
    POTENTIAL_ENERGY_CSV = os.path.join(INPUT_DIR, "potential_energy.csv")
    SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Generated in previous step)
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Submission Path
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Statistics File (Mean/Std for targets)
    STATS_FILE = os.path.join(PROCESSED_DIR, "stats.npy")

    # ==========================================
    # Data Processing Constants
    # ==========================================
    # Graph Construction
    MAX_RADIUS = 5.0  # Angstroms (Radius graph cutoff)

    # Atom Mapping (H, C, N, O, F)
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # Coupling Type Mapping
    COUPLING_TYPES = ["1JHC", "2JHC", "3JHC", "1JHN", "2JHN", "3JHN", "2JHH", "3JHH"]
    COUPLING_TYPE_MAP = {t: i for i, t in enumerate(COUPLING_TYPES)}
    NUM_COUPLING_TYPES = len(COUPLING_TYPES)

    # SoA Array Filenames (Templates)
    # These keys are used to organize the flattened arrays in the processed directory
    SOA_ARRAYS = {
        "mol_id": "mol_id.npy",  # Molecule ID for each node
        "node_type": "node_type.npy",  # Atomic number/type index
        "pos": "pos.npy",  # XYZ coordinates
        "edge_index": "edge_index.npy",  # Radius graph edges (2, E)
        "edge_vec": "edge_vec.npy",  # Edge vectors (E, 3)
        "edge_dist": "edge_dist.npy",  # Edge distances (E,)
        "triplet_index": "triplet_index.npy",  # Triplet indices (3, T) for angles
        # Targets & Indices
        "coupling_atom_index": "coupling_atom_index.npy",  # (2, C) indices of coupling pairs
        "coupling_type": "coupling_type.npy",  # (C,) type index
        "coupling_value": "coupling_value.npy",  # (C,) target value
        "coupling_id": "coupling_id.npy",  # (C,) original submission ID
        # Auxiliary Targets
        "aux_charge": "aux_charge.npy",  # (N,) Mulliken charges
        "aux_shielding": "aux_shielding.npy",  # (N, 9) Shielding tensors
    }

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Architecture
    HIDDEN_DIM = 256
    NUM_LAYERS = 4  # Number of interaction blocks
    NUM_RBF = 64  # Number of RBFs for distance expansion
    NUM_ANGLE_RBF = 32  # Number of RBFs for angle expansion
    RBF_GAMMA = 10.0  # Spread of Gaussian RBF
    DROPOUT = 0.0
    ACTIVATION = "silu"  # Swish activation

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    # Optimization
    BATCH_SIZE = 1024  # Large batch size for A100
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-6
    MAX_EPOCHS = 30  # Sufficient for convergence
    WARMUP_EPOCHS = 2  # Linear warmup

    # Loss Weighting
    LAMBDA_AUX = 0.1  # Weight for auxiliary losses (Charge/Shielding)

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 12  # Number of data loading workers (vCPUs)

    @classmethod
    def get_processed_file_path(cls, split, array_name):
        """
        Helper to generate the full path for a processed array file.
        Args:
            split (str): 'train', 'val', or 'test'
            array_name (str): Key from SOA_ARRAYS (e.g., 'pos')
        Returns:
            str: Full file path
        """
        filename = f"{split}_{cls.SOA_ARRAYS[array_name]}"
        return os.path.join(cls.PROCESSED_DIR, filename)

    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for reproducibility across libraries.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
