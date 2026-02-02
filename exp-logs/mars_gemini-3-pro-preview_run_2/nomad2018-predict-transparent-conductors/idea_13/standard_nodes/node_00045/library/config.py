import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of data loading workers

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directories (Idea specific)
    WORKING_DIR = "./working/idea_14"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Graph Construction
    # -------------------------------------------------------------------------
    CUTOFF_RADIUS = 5.0  # Angstroms
    MAX_NEIGHBORS = 50  # Max neighbors per node to consider

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Node features
    ATOM_EMBEDDING_DIM = 128

    # Edge features (RBF)
    NUM_RBF_BINS = 60
    RBF_MIN = 0.0
    RBF_MAX = 5.0

    # Message Passing
    HIDDEN_DIM = 128
    NUM_LAYERS = 4

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 48
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.1
    NUM_EPOCHS = 200
    EARLY_STOPPING_PATIENCE = 20

    # -------------------------------------------------------------------------
    # Targets
    # -------------------------------------------------------------------------
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    @staticmethod
    def prepare_directories():
        """Creates necessary output directories if they don't exist."""
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
