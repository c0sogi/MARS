import os
import torch


class Config:
    """
    Central configuration for the Angle-Aware Gated Graph Network project.
    Contains hyperparameters for data processing, model architecture, and training.
    """

    # -------------------------------------------------------------------------
    # Paths and Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Working directory for caching processed graphs and model checkpoints
    CACHE_DIR = "./working/idea_4/"

    # Ensure the cache directory exists
    os.makedirs(CACHE_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Cutoff radius for neighbor search in Angstroms (Cite solution_lesson_node_00007)
    CUTOFF_RADIUS = 5.0

    # Maximum number of neighbors to consider per atom to bound computational cost
    MAX_NEIGHBORS = 12

    # Number of bins for Radial Basis Function (RBF) expansion of distances (Cite solution_lesson_node_00005)
    RBF_NUM_BINS = 60

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Dimension of node (atom), edge (bond), and angle embeddings (Cite solution_lesson_node_00004)
    EMBEDDING_DIM = 64

    # Number of message passing blocks (alternating Line Graph and Atom Graph updates)
    # (Cite solution_lesson_node_00006)
    NUM_BLOCKS = 4

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Batch size for training and evaluation (Cite solution_lesson_node_00005)
    BATCH_SIZE = 24

    # Initial learning rate for the AdamW optimizer
    LEARNING_RATE = 1e-3

    # Weight decay for regularization (Cite solution_lesson_node_00003)
    WEIGHT_DECAY = 1e-4

    # Total number of training epochs
    NUM_EPOCHS = 100

    # Patience for Early Stopping (number of epochs with no improvement in validation loss)
    PATIENCE = 15

    # Factor by which to reduce learning rate when validation loss plateaus
    SCHEDULER_FACTOR = 0.5

    # Patience for the learning rate scheduler
    SCHEDULER_PATIENCE = 5

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    # Fixed random seed for reproducibility
    SEED = 42

    # Target columns to predict from the dataset
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # Computation device (GPU if available, else CPU)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def print_config(cls):
        """Prints the current configuration settings."""
        print("=" * 40)
        print("CONFIG")
        print("=" * 40)
        for k, v in cls.__dict__.items():
            if not k.startswith("__") and not callable(v):
                print(f"{k:<20}: {v}")
        print("=" * 40)
