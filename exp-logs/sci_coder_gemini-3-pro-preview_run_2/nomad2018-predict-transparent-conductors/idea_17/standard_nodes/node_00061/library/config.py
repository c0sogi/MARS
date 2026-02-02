import os


class Config:
    """
    Configuration class for the Learnable-Residual Crystal Graph Network (LR-CGCNN) project.
    """

    # -------------------------------------------------------------------------
    # Global Settings
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Directory Structure
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_17"

    # Sub-directories for organized output
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

    # Ensure necessary directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Metadata inputs
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cached data files (using .npz for efficient storage of graph data lists)
    TRAIN_GRAPHS_CACHE = os.path.join(CACHE_DIR, "train_graphs.npz")
    VAL_GRAPHS_CACHE = os.path.join(CACHE_DIR, "val_graphs.npz")
    TEST_GRAPHS_CACHE = os.path.join(CACHE_DIR, "test_graphs.npz")

    # Scaler cache
    TARGET_SCALER_CACHE = os.path.join(CACHE_DIR, "target_scaler.npz")

    # Output files
    MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    RADIUS = 5.0  # Cutoff radius for neighbor search in Angstroms
    MAX_NUM_NBR = 12  # Maximum number of neighbors to consider per atom

    # -------------------------------------------------------------------------
    # Model Architecture Parameters
    # -------------------------------------------------------------------------
    ATOM_FEA_LEN = 128  # Dimension of node (atom) embeddings
    N_CONV = 4  # Number of GNN interaction layers
    H_FEA_LEN = 128  # Hidden dimension within interaction blocks
    N_H = 1  # Number of hidden layers in the output MLP heads
    N_RBF = 60  # Number of Gaussian RBF bins for edge distance expansion

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 48
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.1
    EPOCHS = 150
    PATIENCE = 20  # Early stopping patience (epochs without improvement)

    # -------------------------------------------------------------------------
    # Target Configuration
    # -------------------------------------------------------------------------
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    DEBUG = False  # Set to True to run on a small subset of data
    MAX_SAMPLES = 100  # Number of samples to use when DEBUG is True
