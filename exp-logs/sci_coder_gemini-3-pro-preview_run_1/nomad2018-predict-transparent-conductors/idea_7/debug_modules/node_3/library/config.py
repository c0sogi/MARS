import os


class Config:
    """
    Configuration class for the Symmetry-Informed Residual Deep Sets (SI-RDS) strategy.
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
    WORKING_DIR = "./working/idea_7"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Geometry Data Root
    GEOMETRY_DIR = INPUT_DIR

    # Cache Files (for deterministic data processing)
    TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_CACHE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_CACHE = os.path.join(WORKING_DIR, "test_data.npz")

    # Model Checkpoint
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pt")

    # Final Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Maximum number of atoms to pad/truncate to (based on dataset analysis max=80)
    MAX_ATOMS = 80

    # Debugging: set to a small integer (e.g., 100) to limit dataset size, or None for full data
    DEBUG_SAMPLE_SIZE = None

    # Feature Dimensions
    # Atomic Stream: One-hot (4) + Centered Coords (3) + PBC Neighbor Dist (1)
    ATOMIC_FEATURE_DIM = 8

    # Global Stream: Lattice (6) + Volume (1) + Density (1) + Stoichiometry (3)
    GLOBAL_FEATURE_DIM = 11

    # Symmetry Stream: Spacegroup ID (1-230)
    NUM_SPACEGROUPS = 231

    # Targets
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    NUM_TARGETS = 2

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Dimension of the symmetry embedding
    SYMMETRY_EMBED_DIM = 16

    # Hidden dimension for MLPs and Residual Blocks
    HIDDEN_DIM = 128

    # Number of residual blocks in the Atomic Stream
    NUM_RES_BLOCKS = 3

    # Dropout rate for regularization
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    EPOCHS = 150
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Early Stopping
    PATIENCE = 15

    # Learning Rate Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-6

    def __init__(self):
        """
        Initialize configuration and ensure necessary directories exist.
        """
        os.makedirs(self.WORKING_DIR, exist_ok=True)
        os.makedirs(self.SUBMISSION_DIR, exist_ok=True)
