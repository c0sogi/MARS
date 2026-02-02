import os
import torch
from dataclasses import dataclass

# ==========================================
# Global Constants & Mappings
# ==========================================

# Canonical Integer Mapping for Atom Types
# Based on QM9/CHamps dataset composition
ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}

# Canonical Integer Mapping for Scalar Coupling Types
TYPE_MAP = {
    "1JHC": 0,
    "2JHC": 1,
    "3JHC": 2,
    "1JHN": 3,
    "2JHN": 4,
    "3JHN": 5,
    "2JHH": 6,
    "3JHH": 7,
}

# Inverse mapping for decoding predictions or logging
INVERSE_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}


@dataclass
class Config:
    """
    Configuration for the Molecule-Parallel Deep Interaction Network (MP-DIN).
    """

    # ==============================
    # File System Paths
    # ==============================
    INPUT_DIR: str = "./input"
    METADATA_DIR: str = "./metadata"
    # Working directory for Idea 19 (MP-DIN)
    WORKING_DIR: str = "./working/idea_19"
    SUBMISSION_DIR: str = "./submission"

    # Input Data Files
    STRUCTURES_CSV: str = os.path.join(INPUT_DIR, "structures.csv")
    TRAIN_METADATA: str = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA: str = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA: str = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Files
    SUBMISSION_PATH: str = os.path.join(SUBMISSION_DIR, "submission.csv")
    MODEL_SAVE_PATH: str = os.path.join(WORKING_DIR, "best_model.pth")
    STATS_PATH: str = os.path.join(WORKING_DIR, "stats.npy")

    # ==============================
    # Model Hyperparameters
    # ==============================
    # Backbone: Deep Node-Centric Continuous Filter Network
    node_dim: int = 128  # Dimension of atom embeddings
    num_layers: int = 6  # Number of interaction blocks (depth)
    num_rbf: int = 128  # Number of Gaussian RBF centers
    rbf_min: float = 0.0  # Min distance for RBF
    rbf_max: float = 15.0  # Max distance for RBF (covering large molecules)
    cutoff: float = 5.0  # Graph connectivity radius (Angstroms)

    # Readout: Interaction-Aware Shared Conditional Head
    type_embed_dim: int = 32  # Embedding dimension for coupling type

    # ==============================
    # Training Hyperparameters
    # ==============================
    seed: int = 42
    # Batch size is in MOLECULES.
    # 48 molecules typically contain ~1500-3000 coupling pairs.
    batch_size: int = 48
    epochs: int = 35
    learning_rate: float = 5e-4  # Initial LR for AdamW
    weight_decay: float = 1e-6

    # Scheduler: Cosine Annealing Warm Restarts
    T_max: int = 35  # Cycle length matching epochs
    eta_min: float = 1e-6  # Minimum LR

    # Early Stopping
    patience: int = 6  # Stop if val loss doesn't improve

    # ==============================
    # System & Runtime
    # ==============================
    num_workers: int = 8  # Data loading workers
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ==============================
    # Debugging / Development
    # ==============================
    debug: bool = False  # Set to True to run on a small subset
    debug_samples: int = 2000  # Number of molecules to use in debug mode

    def __post_init__(self):
        """
        Ensure necessary directories exist upon initialization.
        """
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
