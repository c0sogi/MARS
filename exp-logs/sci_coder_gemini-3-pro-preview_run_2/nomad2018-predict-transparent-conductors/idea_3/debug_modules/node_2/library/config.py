import os
import torch


class Config:
    """
    Configuration class for the Distance-Biased Graph Transformer (DB-GT) project.
    Contains file paths, data processing parameters, model architecture settings,
    and training hyperparameters.
    """

    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_3"

    # Cache directory for processed graph data (parquet/npz files)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")

    # Checkpoint directory for saving model weights
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Submission directory
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Metadata file paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # -------------------------------------------------------------------------
    # Data Processing Parameters
    # -------------------------------------------------------------------------
    # Cutoff radius for constructing the neighbor graph (in Angstroms)
    CUTOFF_RADIUS = 5.0

    # Number of bins for the Gaussian Radial Basis Function (RBF) expansion of edge distances
    RBF_BINS = 60

    # Maximum number of neighbors per node to consider (for efficiency)
    MAX_NEIGHBORS = 50

    # Atomic species in the dataset: O (8), Al (13), Ga (31), In (49)
    # We set the embedding table size to accommodate atomic numbers up to In (49) comfortably.
    MAX_ATOMIC_NUMBER = 100

    # Target columns to predict
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters
    # -------------------------------------------------------------------------
    # Dimension of node embeddings and hidden states
    EMBEDDING_DIM = 64

    # Number of Distance-Biased Transformer Layers
    N_LAYERS = 4

    # Number of attention heads in the Multi-Head Attention mechanism
    NUM_HEADS = 4

    # Dropout rate for regularization
    DROPOUT = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 48
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 100
    PATIENCE = 10  # Early stopping patience (epochs without improvement)

    # -------------------------------------------------------------------------
    # System / Reproducibility / Debugging
    # -------------------------------------------------------------------------
    SEED = 42
    NUM_WORKERS = 4  # Number of subprocesses for data loading
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Debugging flags
    DEBUG = False  # Set to True to run on a small subset of data
    DEBUG_SIZE = 100  # Number of samples to use when DEBUG is True

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
