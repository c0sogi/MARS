import os


class Config:
    """
    Configuration constants and hyperparameters for the Composition-Aware CGCNN project.
    """

    # --------------------------------------------------------------------------
    # Directory and File Paths
    # --------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_8"

    # Subdirectories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Ensure necessary writeable directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Sample submission for format reference
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # --------------------------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------------------------
    SEED = 42

    # --------------------------------------------------------------------------
    # Data Processing Parameters
    # --------------------------------------------------------------------------
    # Cutoff radius for neighbor search in Angstroms
    RADIUS = 5.0

    # Maximum number of neighbors to consider per atom (for efficiency)
    MAX_NEIGHBORS = 12

    # Atom types present in the dataset (Al, Ga, In, O)
    ATOM_TYPES = ["Al", "Ga", "In", "O"]

    # --------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # --------------------------------------------------------------------------
    # Dimension of atom embeddings (node features)
    ATOM_FEA_LEN = 128

    # Number of Graph Convolution layers
    N_CONV = 4

    # Number of bins for Gaussian Radial Basis Functions (edge features)
    RBF_N_BINS = 80

    # Dropout rate used in the model
    DROPOUT = 0.2

    # Dimension of global features (6 lattice params + 4 composition fractions)
    GLOBAL_FEA_LEN = 10

    # --------------------------------------------------------------------------
    # Training Hyperparameters
    # --------------------------------------------------------------------------
    BATCH_SIZE = 48

    # Learning Rate
    LR = 1e-3

    # Weight Decay for regularization (AdamW)
    WEIGHT_DECAY = 1e-4

    # Maximum number of training epochs
    EPOCHS = 150

    # Patience for Early Stopping
    PATIENCE = 25

    # Target columns
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
