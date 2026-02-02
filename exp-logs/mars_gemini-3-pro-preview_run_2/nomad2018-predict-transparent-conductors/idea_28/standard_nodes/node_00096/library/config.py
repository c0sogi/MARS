import os

# Create the working directory for this idea if it doesn't exist
WORKING_DIR = "./working/idea_29"
os.makedirs(WORKING_DIR, exist_ok=True)


class Config:
    """
    Configuration class for the Stabilized Receiver-Aware Crystal Graph Network (S-RA-CGN).
    """

    # -------------------------------------------------------------------------
    # General Settings
    # -------------------------------------------------------------------------
    SEED = 42
    DEBUG = False  # Set to True to run on a subset of data for debugging

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = WORKING_DIR

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache file paths
    TRAIN_GRAPH_CACHE = os.path.join(WORKING_DIR, "train_graphs.npz")
    VAL_GRAPH_CACHE = os.path.join(WORKING_DIR, "val_graphs.npz")
    TEST_GRAPH_CACHE = os.path.join(WORKING_DIR, "test_graphs.npz")
    TARGET_SCALER_CACHE = os.path.join(WORKING_DIR, "target_scaler.npz")

    # Model checkpoint path
    MODEL_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

    # Submission path
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing / Graph Construction
    # -------------------------------------------------------------------------
    # Cutoff radius for neighbor search in Angstroms
    CUTOFF_RADIUS = 5.0
    # Maximum number of neighbors to consider (for fixed-size tensor construction)
    MAX_NEIGHBORS = 50

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # Dimension of atomic number embeddings
    NODE_EMBED_DIM = 128
    # Number of Gaussian Radial Basis Function bins for edge encoding
    RBF_BINS = 60
    # Hidden dimension size used throughout the interaction blocks
    HIDDEN_DIM = 128
    # Number of interaction blocks in the network
    NUM_INTERACTION_BLOCKS = 4
    # Dropout rate applied in interaction blocks and readout
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Batch size for training and validation
    BATCH_SIZE = 48
    # Initial learning rate for AdamW optimizer
    LEARNING_RATE = 1e-3
    # Weight decay for regularization
    WEIGHT_DECAY = 1e-4
    # Maximum number of training epochs
    NUM_EPOCHS = 150
    # Patience for early stopping (number of epochs without improvement)
    EARLY_STOPPING_PATIENCE = 15

    # -------------------------------------------------------------------------
    # Scheduler Hyperparameters (ReduceLROnPlateau)
    # -------------------------------------------------------------------------
    SCHEDULER_FACTOR = 0.6
    SCHEDULER_PATIENCE = 5
    SCHEDULER_MIN_LR = 1e-6

    # -------------------------------------------------------------------------
    # Target Variables
    # -------------------------------------------------------------------------
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    @classmethod
    def print_config(cls):
        """Prints the current configuration."""
        print("=" * 40)
        print("CONFIG")
        print("=" * 40)
        for attr in dir(cls):
            if not attr.startswith("__") and not callable(getattr(cls, attr)):
                print(f"{attr:<25}: {getattr(cls, attr)}")
        print("=" * 40)
