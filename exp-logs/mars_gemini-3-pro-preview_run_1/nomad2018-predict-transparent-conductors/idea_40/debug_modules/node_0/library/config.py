import os
import torch


class Config:
    """
    Configuration for Density-Calibrated Chemically-Contextualized Wide Deep Sets (DC3-WDS).
    """

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORKING_DIR = "./working/idea_40"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    SUBMISSION_DIR = "./submission"

    # File paths for metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

    # Output paths
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pt")

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Data Processing
    # -------------------------------------------------------------------------
    # Atom mapping for One-Hot Encoding
    ATOM_MAP = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # Neighbor search parameters for local context
    K_NEIGHBORS = 12

    # Feature Dimensions
    # Atomic Stream Features:
    # 1. Identity (One-Hot): 4
    # 2. Centered Coords (x,y,z): 3
    # 3. Min Neighbor Distance (d_min): 1
    # 4. Mean Neighbor Distance (d_mean): 1
    # 5. Soft Chemical Context (weighted composition): 4
    # Total: 13
    ATOMIC_FEATURE_DIM = 13

    # Global Stream Features:
    # 1. Lattice Vector Lengths (a,b,c): 3
    # 2. Lattice Angles (alpha, beta, gamma): 3
    # 3. Unit Cell Volume: 1
    # 4. Atomic Density: 1
    # 5. Stoichiometry (percent_al, percent_ga, percent_in): 3
    # 6. Total Atoms: 1
    # Total: 12
    GLOBAL_FEATURE_DIM = 12

    # Target Variables
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
    NUM_TARGETS = len(TARGET_COLS)

    # -------------------------------------------------------------------------
    # Model Architecture (DC3-WDS)
    # -------------------------------------------------------------------------
    # Atomic Stream (Wide MLP)
    ATOMIC_HIDDEN_DIM = 512
    ATOMIC_LAYERS = 3

    # Global Stream (High-Capacity MLP)
    GLOBAL_HIDDEN_DIM = 256
    GLOBAL_LAYERS = 2

    # Fusion Head
    # Note: Latent dim will be 2 * ATOMIC_HIDDEN_DIM (due to Mean+Max pooling) + GLOBAL_HIDDEN_DIM

    # Regularization
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 200
    PATIENCE = 20  # Early stopping patience

    # Scheduler settings (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5

    # -------------------------------------------------------------------------
    # Debugging / Development
    # -------------------------------------------------------------------------
    # Set to True to run on a small subset of data for quick pipeline verification
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4

    @staticmethod
    def setup():
        """Ensure necessary working directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories on module import
Config.setup()
