import os


class Config:
    """
    Configuration for Potential-Augmented Wide Deep Sets (PA-WDS).
    Defines file paths, model hyperparameters, training settings, and feature engineering constants.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_12"
    SUBMISSION_DIR = "./submission"

    # Input CSVs (using metadata splits)
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    CACHE_PATH_TRAIN = os.path.join(WORKING_DIR, "train_data.npz")
    CACHE_PATH_VAL = os.path.join(WORKING_DIR, "val_data.npz")
    CACHE_PATH_TEST = os.path.join(WORKING_DIR, "test_data.npz")

    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing & Feature Engineering
    # -------------------------------------------------------------------------
    # Set to an integer (e.g., 100) to limit dataset size for debugging, or None for full data
    DEBUG_SAMPLE_SIZE = None

    # Potential Proxy
    POTENTIAL_K = 12  # Number of nearest neighbors for inverse distance summation

    # Atomic Identity
    ATOM_TYPES = ["Al", "Ga", "In", "O"]
    ATOM_TO_IDX = {atom: i for i, atom in enumerate(ATOM_TYPES)}
    NUM_ATOM_TYPES = len(ATOM_TYPES)

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Atomic Stream (Wide Point Processor)
    # Features: One-hot (4) + Coords (3) + NN Dist (1) + Potential (1) = 9
    ATOMIC_INPUT_DIM = 9
    ATOMIC_HIDDEN_DIM = 512
    ATOMIC_DROPOUT = 0.2  # Regularization for wide layers

    # Global Stream (Thermodynamic Context)
    # Features: Lattice lengths (3) + Angles (3) + Volume (1) + Density (1) + Composition (3) = 11
    GLOBAL_INPUT_DIM = 11
    GLOBAL_HIDDEN_DIM = 256
    GLOBAL_NUM_LAYERS = 3
    GLOBAL_DROPOUT = 0.1

    # Fusion Head
    # Input: (Atomic_Hidden * 2 for Mean+Max Pool) + Global_Hidden
    FUSION_HIDDEN_DIM = 128
    OUTPUT_DIM = 2  # formation_energy, bandgap_energy

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # L2 Regularization
    NUM_EPOCHS = 200
    PATIENCE = 20  # Early stopping patience

    # Targets
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
