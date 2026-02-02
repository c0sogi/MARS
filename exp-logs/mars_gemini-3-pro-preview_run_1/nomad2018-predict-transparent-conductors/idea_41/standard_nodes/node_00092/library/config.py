import os


class Config:
    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42

    # Paths
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_41"
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

    # Cache Files (Parquet/NPZ)
    TRAIN_DATA_CACHE = os.path.join(WORKING_DIR, "train_data.npz")
    VAL_DATA_CACHE = os.path.join(WORKING_DIR, "val_data.npz")
    TEST_DATA_CACHE = os.path.join(WORKING_DIR, "test_data.npz")
    SCALERS_CACHE = os.path.join(WORKING_DIR, "scalers.npz")

    # Model Checkpoint
    MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pt")

    # Submission Output
    SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Atomic Species for One-Hot Encoding
    ATOMIC_SPECIES = ["Al", "Ga", "In", "O"]
    NUM_SPECIES = len(ATOMIC_SPECIES)

    # Neighbor Search Parameters (Multi-Scale)
    K_SHORT = 6  # Immediate coordination environment
    K_MED = 24  # Medium-range crystal field environment

    # Feature Dimensions
    # Atomic Features:
    #   4 (One-Hot) + 3 (Centered Coords) + 1 (NN Dist) +
    #   4 (Short Context) + 4 (Med Context) = 16
    ATOMIC_INPUT_DIM = 16

    # Global Features:
    #   3 (Lattice Lens) + 3 (Lattice Angles) + 1 (Volume) +
    #   1 (Density) + 3 (Stoichiometry) + 1 (Total Atoms) = 12
    GLOBAL_INPUT_DIM = 12

    # Targets
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    NUM_TARGETS = len(TARGET_COLS)

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Atomic Stream (Wide MLP)
    ATOMIC_HIDDEN_DIM = 512
    ATOMIC_LAYERS = 3

    # Global Stream
    GLOBAL_HIDDEN_DIM = 256
    GLOBAL_LAYERS = 2

    # Regularization
    DROPOUT_RATE = 0.1
    USE_BATCH_NORM = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 32  # Number of crystals per batch (Sparse Batching)
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4  # Regularization for wide layers
    NUM_EPOCHS = 200
    PATIENCE = 20  # Early stopping patience

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # Set to None to use full dataset, or an integer to limit samples for quick testing
    DEBUG_SAMPLE_SIZE = None

    @staticmethod
    def ensure_directories():
        """Creates necessary working directories."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
