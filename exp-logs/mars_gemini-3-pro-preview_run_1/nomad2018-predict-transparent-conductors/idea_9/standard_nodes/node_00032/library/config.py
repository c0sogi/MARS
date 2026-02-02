import os


class Config:
    """
    Configuration class for the Symmetry-Informed Residual Deep Sets with Statistical Pooling (SI-RDS-SP) strategy.
    Acts as a single source of truth for paths, hyperparameters, and constants.
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Writable working directory for caching and model checkpoints
    WORKING_DIR = "./working/idea_9"

    # Submission directory
    SUBMISSION_DIR = "./submission"

    # Specific file paths
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Cache files for processed data (using .npz for efficient numpy storage)
    TRAIN_DATA_CACHE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_DATA_CACHE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_DATA_CACHE = os.path.join(WORKING_DIR, "test_data.npz")

    # Model checkpoint path
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")

    # Final submission file
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Parameters
    # -------------------------------------------------------------------------
    # Atom types for one-hot encoding
    ATOM_TYPES = ["Al", "Ga", "In", "O"]
    NUM_ATOM_TYPES = len(ATOM_TYPES)

    # Spacegroups range from 1 to 230
    MAX_SPACEGROUP = 230

    # Input Feature Dimensions
    # Atomic features: One-hot (4) + Coords (3) + NN Dist (1)
    ATOMIC_INPUT_DIM = 8

    # Global features:
    # Lattice lengths (3) + Angles (3) + Stoichiometry (3) + Total Atoms (1) + Volume (1) + Density (1)
    GLOBAL_INPUT_DIM = 12

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Atomic Stream (Residual Point Processor)
    ATOMIC_HIDDEN_DIM = 128
    ATOMIC_RES_BLOCKS = 3  # Number of residual blocks
    ATOMIC_DROPOUT = 0.1

    # Global Stream (Thermodynamic Context)
    GLOBAL_HIDDEN_DIM = 128
    GLOBAL_LAYERS = 3
    GLOBAL_DROPOUT = 0.1

    # Symmetry Stream (Crystallographic Prior)
    SYMMETRY_EMBEDDING_DIM = 32

    # Fusion Head
    # Concatenated dim: (ATOMIC_HIDDEN_DIM * 3 for Mean/Max/Std) + GLOBAL_HIDDEN_DIM + SYMMETRY_EMBEDDING_DIM
    FUSION_HIDDEN_DIM = 256
    FUSION_LAYERS = 3
    FUSION_DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Parameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    PATIENCE = 20  # Early stopping patience
    SCHEDULER_FACTOR = 0.5  # ReduceLROnPlateau factor
    SCHEDULER_PATIENCE = 5  # ReduceLROnPlateau patience
    MIN_LR = 1e-6

    @classmethod
    def setup(cls):
        """Ensures all necessary writable directories exist."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
Config.setup()
