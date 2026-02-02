import os


class Config:
    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    # Specific working directory for this idea to avoid conflicts
    WORKING_DIR = "./working/idea_26"

    # Input Metadata
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Caching Paths (using .npz for efficient storage of graph data)
    # These paths are used by the data processing module to load/save deterministic data
    TRAIN_GRAPHS_CACHE = os.path.join(WORKING_DIR, "cache", "train_graphs.npz")
    VAL_GRAPHS_CACHE = os.path.join(WORKING_DIR, "cache", "val_graphs.npz")
    TEST_GRAPHS_CACHE = os.path.join(WORKING_DIR, "cache", "test_graphs.npz")
    TARGET_SCALER_CACHE = os.path.join(WORKING_DIR, "cache", "target_scaler.npz")

    # Output Paths
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    # Reproducibility
    SEED = 42

    # Graph Construction
    CUTOFF = 5.0  # Radius graph cutoff in Angstroms
    MAX_NEIGHBORS = (
        12  # Maximum neighbors per node (Cite Lesson 00081: Radius Graph capped at 12)
    )

    # Model Architecture (RA-GLU-Net)
    HIDDEN_DIM = 128  # Dimension of node and edge embeddings
    NUM_LAYERS = 4  # Number of interaction blocks
    NUM_RBF = 60  # Number of Gaussian Radial Basis Functions
    ATOM_EMBEDDING_DIM = 128
    MAX_ATOMIC_NUMBER = 100

    # Training
    BATCH_SIZE = 48
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 300

    # Optimization & Scheduling
    PATIENCE = 20  # Early stopping patience
    SCHEDULER_PATIENCE = 10
    SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6

    # Targets
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    @classmethod
    def setup(cls):
        """
        Creates the necessary working directories for cache, checkpoints, and submissions.
        """
        os.makedirs(os.path.dirname(cls.TRAIN_GRAPHS_CACHE), exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        print(f"Directories created/verified at: {cls.WORKING_DIR}")
