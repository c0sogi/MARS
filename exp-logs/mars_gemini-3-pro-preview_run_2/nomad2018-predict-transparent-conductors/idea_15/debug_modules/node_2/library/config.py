import os
import torch


class Config:
    """
    Configuration class for Hybrid Amplified-Structural and Compositional Network (HASC-Net).
    """

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_15"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Radius for constructing the crystal graph (in Angstroms)
    CUTOFF_RADIUS = 5.0

    # Maximum number of neighbors to consider per node (for efficiency)
    MAX_NEIGHBORS = 50

    # Random seed for reproducibility
    SEED = 42

    # Debugging: Set to an integer to limit the number of samples (e.g., 100)
    # Set to None to use the full dataset.
    DEBUG_DATA_LIMIT = None

    # Target columns
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # -------------------------------------------------------------------------
    # Model Architecture Parameters
    # -------------------------------------------------------------------------
    # Dimension of atomic number embeddings (covering up to Uranium usually suffices)
    ATOM_INPUT_DIM = 92

    # Hidden dimension size for GNN and MLP layers
    HIDDEN_DIM = 128

    # Number of message passing layers in the GNN backbone
    NUM_LAYERS = 4

    # Dropout rate
    DROPOUT = 0.1

    # Number of Gaussian Radial Basis Functions for edge encoding
    N_RBF = 50

    # Scaling factor for the Amplified Residual connection (h_new = 2*h_old + message)
    RESIDUAL_SCALE = 2.0

    # Dimension of the global feature vector
    # 3 (Al, Ga, In composition) + 6 (Lattice parameters a, b, c, alpha, beta, gamma)
    GLOBAL_FEATURE_DIM = 9

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Batch size for training and evaluation
    BATCH_SIZE = 48

    # Initial learning rate for AdamW optimizer
    LEARNING_RATE = 1e-3

    # Weight decay for regularization
    WEIGHT_DECAY = 1e-4

    # Maximum number of training epochs
    NUM_EPOCHS = 300

    # Patience for early stopping (number of epochs without improvement)
    PATIENCE = 20

    # -------------------------------------------------------------------------
    # Compute Device
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print("CONFIG")
        print("=" * 40)
        for attr in dir(cls):
            if not attr.startswith("__") and not callable(getattr(cls, attr)):
                print(f"{attr:<20}: {getattr(cls, attr)}")
        print("=" * 40)
