import os
import torch


class Config:
    """
    Configuration class for the Lattice-Informed Crystal Graph Convolutional Network
    with Element-wise Learnable Residuals (LI-CGCNN-ELR).
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # File System Paths
    # -------------------------------------------------------------------------
    # Input Data Directories (Read-Only)
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata CSV Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directory (Write Allowed)
    # Using 'idea_18' as the specific experiment identifier
    WORKING_DIR = "./working/idea_18"

    # Cache Directory for Deterministic Data Processing
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Cached File Paths
    TRAIN_GRAPHS_CACHE = os.path.join(CACHE_DIR, "train_graphs.npz")
    VAL_GRAPHS_CACHE = os.path.join(CACHE_DIR, "val_graphs.npz")
    TEST_GRAPHS_CACHE = os.path.join(CACHE_DIR, "test_graphs.npz")

    # Scaler Cache Paths
    TARGET_SCALER_PATH = os.path.join(CACHE_DIR, "target_scaler.npz")
    LATTICE_SCALER_PATH = os.path.join(CACHE_DIR, "lattice_scaler.npz")

    # Checkpoint Directory for Model Weights
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Submission Directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Graph Construction
    CUTOFF_RADIUS = 5.0  # Angstroms (Strict cutoff for local interactions)
    MAX_NEIGHBORS = 12  # Maximum number of neighbors to consider per node

    # Target Variables
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Local Graph Backbone (CGCNN)
    ATOM_EMBEDDING_DIM = 128
    NUM_CGCONV_LAYERS = 4

    # Edge Encoding (Gaussian RBF)
    RBF_BINS = 60
    RBF_MIN = 0.0
    RBF_MAX = 5.0  # Matches CUTOFF_RADIUS

    # Lattice Context Stream
    LATTICE_INPUT_DIM = 6  # [a, b, c, alpha, beta, gamma]
    LATTICE_EMBEDDING_DIM = 32  # Dimension for global lattice context embedding

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 48  # Selected to introduce beneficial gradient noise
    LEARNING_RATE = 1e-3  # Standard starting learning rate
    WEIGHT_DECAY = 1e-4  # Regularization strength
    DROPOUT_RATE = 0.1  # Dropout probability within interaction blocks
    NUM_EPOCHS = 150  # Maximum training epochs
    PATIENCE = 15  # Early stopping patience (epochs without improvement)

    # Hardware Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of subprocesses for data loading

    @staticmethod
    def setup_directories():
        """
        Creates the necessary working directories for cache, checkpoints, and submission.
        This ensures directory safety before any file operations.
        """
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories initialized at: {Config.WORKING_DIR}")
