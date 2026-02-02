import os


class Config:
    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    # Read-only input directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working directory for artifacts (cache, checkpoints, etc.)
    # We use a specific subdirectory for this idea iteration
    WORKING_DIR = "./working/idea_3"

    # Subdirectories for organization
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    RANDOM_SEED = 42

    # -------------------------------------------------------------------------
    # Data / Graph Construction
    # -------------------------------------------------------------------------
    # Cutoff radius for neighbor finding (in Angstroms)
    # Explicitly set to 5.0 to avoid noise from distant neighbors
    CUTOFF_RADIUS = 5.0

    # Maximum number of neighbors to consider per atom (for efficiency)
    MAX_NEIGHBORS = 50

    # Target columns to predict
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # -------------------------------------------------------------------------
    # Model Architecture (CGCNN)
    # -------------------------------------------------------------------------
    # Dimension of node and edge embeddings
    HIDDEN_DIM = 128

    # Number of message passing interaction layers
    NUM_LAYERS = 4

    # Number of attention heads (Not used in CGCNN, kept for compatibility if needed)
    NUM_HEADS = 4

    # Number of Gaussian basis functions for edge distance expansion
    NUM_RBF = 50

    # Dropout rate for regularization within message passing blocks
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    # Initial learning rate
    LEARNING_RATE = 1e-3

    # Weight decay for regularization (AdamW)
    WEIGHT_DECAY = 1e-4

    # Batch size
    BATCH_SIZE = 48

    # Maximum number of training epochs
    NUM_EPOCHS = 100

    # Early stopping patience (number of epochs with no improvement)
    PATIENCE = 10

    # Factor to reduce learning rate by when validation loss plateaus
    SCHEDULER_FACTOR = 0.5

    # Patience for the scheduler
    SCHEDULER_PATIENCE = 5

    @staticmethod
    def setup():
        """
        Ensures necessary working directories exist.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# Initialize directories immediately upon import
Config.setup()
