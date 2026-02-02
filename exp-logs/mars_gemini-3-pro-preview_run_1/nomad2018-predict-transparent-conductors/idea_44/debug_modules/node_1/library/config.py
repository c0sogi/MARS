import os


class Config:
    """
    Configuration class for the Anisotropic Multi-Scale Physics-Aware Deep Sets (AMSP-DS) strategy.
    """

    # ==========================================
    # Directories and File Paths
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_44"  # Cache directory for processed features
    SUBMISSION_DIR = "./submission"

    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Atomic Physics Constants
    # ==========================================
    # Used for composition-weighted physical context
    ATOMIC_MASS = {"Al": 26.981539, "Ga": 69.723, "In": 114.818, "O": 15.999}

    COVALENT_RADIUS = {"Al": 1.21, "Ga": 1.22, "In": 1.42, "O": 0.66}

    ELECTRONEGATIVITY = {"Al": 1.61, "Ga": 1.81, "In": 1.78, "O": 3.44}

    # Mapping for one-hot encoding
    ATOM_MAP = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
    NUM_ATOM_TYPES = 4

    # ==========================================
    # Feature Engineering Hyperparameters
    # ==========================================
    # Neighbor counts for multi-scale context
    K_MIN = 1  # Closest neighbor for distortion
    K_PACKING = 12  # Packing shell for distortion ratio
    K_SHORT = 6  # Short-range chemical context
    K_MEDIUM = 24  # Medium-range chemical context

    # ==========================================
    # Model Architecture Hyperparameters
    # ==========================================
    # Atomic Stream (Wide MLP)
    ATOMIC_HIDDEN_DIM = 512
    ATOMIC_LAYERS = 3

    # Global Stream
    GLOBAL_HIDDEN_DIM = 256
    GLOBAL_LAYERS = 2

    # Fusion & Output
    LATENT_DIM = 256  # Dimension after pooling atomic features
    DROPOUT = 0.1

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    SEED = 42
    BATCH_SIZE = 64  # Number of crystals per batch (sparse batching)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-6

    # ==========================================
    # Debugging
    # ==========================================
    DEBUG = False
    DEBUG_SIZE = 100  # Number of samples to use if DEBUG is True

    @classmethod
    def setup(cls):
        """Creates necessary working directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
