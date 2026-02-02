import os


class Config:
    # -------------------------------------------------------------------------
    # Directory and File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_22"
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure necessary directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Processing & Feature Engineering
    # -------------------------------------------------------------------------
    # Atomic identity encoding
    ATOM_TYPES = ["Al", "Ga", "In", "O"]

    # Neighborhood calculation parameters
    # K=12 is used to compute the Mean Neighborhood Distance (d_mean_12)
    K_NEIGHBORS = 12

    # Input Feature Dimensions
    # Atomic Stream:
    #   4 (One-hot) + 3 (Centered Coords) + 1 (d_min) + 1 (d_mean_12) = 9
    ATOMIC_INPUT_DIM = 9

    # Global Stream:
    #   3 (Lattice lengths) + 3 (Lattice angles) + 1 (Volume) +
    #   1 (Density) + 3 (Stoichiometry) + 1 (Total Atoms) = 12
    GLOBAL_INPUT_DIM = 12

    # -------------------------------------------------------------------------
    # Model Architecture (MSN-WDS)
    # -------------------------------------------------------------------------
    # Wide MLP for Atomic Stream (Immediate Expansion)
    ATOMIC_HIDDEN_DIM = 512

    # High-Capacity MLP for Global Stream
    GLOBAL_HIDDEN_DIM = 256

    # Latent dimension for embeddings before fusion
    LATENT_DIM = 256

    # Regularization safeguards
    DROPOUT_RATE = 0.2
    USE_BATCH_NORM = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200
    PATIENCE = 20  # Early stopping patience

    # -------------------------------------------------------------------------
    # Target Configuration
    # -------------------------------------------------------------------------
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    # Apply log(1+y) transformation to targets
    LOG_TARGETS = True
