import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # Path Configuration
    # ==========================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_2"
    SUBMISSION_DIR = "./submission"

    # Input Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")

    # Cache Files (using .npz for numpy arrays)
    CACHE_TRAIN_DATA = os.path.join(WORKING_DIR, "cached_train_data.npz")
    CACHE_VAL_DATA = os.path.join(WORKING_DIR, "cached_val_data.npz")
    CACHE_TEST_DATA = os.path.join(WORKING_DIR, "cached_test_data.npz")

    # Output Files
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # Data & Physics Constants
    # ==========================================
    # Atom mapping (H, C, N, O, F are the elements in this dataset)
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # Coupling Types
    COUPLING_TYPES = ["1JHC", "2JHH", "1JHN", "2JHN", "2JHC", "3JHH", "3JHC", "3JHN"]
    COUPLING_TYPE_MAP = {t: i for i, t in enumerate(COUPLING_TYPES)}
    INVERSE_COUPLING_TYPE_MAP = {i: t for t, i in COUPLING_TYPE_MAP.items()}
    NUM_COUPLING_TYPES = len(COUPLING_TYPES)

    # Graph Construction
    CUTOFF_RADIUS = 5.0  # Angstroms, generous cutoff for neighbor lists
    MAX_NEIGHBORS = 32  # Maximum number of neighbors to consider per atom

    # ==========================================
    # Model Hyperparameters (DMPNN)
    # ==========================================
    HIDDEN_DIM = 128  # Dimension of atom/edge embeddings
    NUM_INTERACTIONS = 4  # Number of message passing layers
    NUM_RBF = 32  # Number of Radial Basis Functions
    NUM_ABF = 16  # Number of Angular Basis Functions (for triplets)
    ACTIVATION = "swish"  # Non-linear activation
    DROPOUT = 0.0  # Deterministic model, no dropout

    # ==========================================
    # Training Configuration
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    BATCH_SIZE = 128  # A100 40GB allows for larger batches
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-6
    EPOCHS = 25
    WARMUP_EPOCHS = 2
    PATIENCE = 5  # Early stopping patience

    # Dataloader
    NUM_WORKERS = 4

    # Debugging / Development
    # Set DEBUG_SAMPLE_SIZE to an integer (e.g., 5000) to train on a subset
    DEBUG_SAMPLE_SIZE = None

    @classmethod
    def setup(cls):
        """
        Initialize the working environment:
        1. Create necessary directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @classmethod
    def get_dataset_size(cls, full_size):
        """Helper to determine effective dataset size based on debug flag."""
        if cls.DEBUG_SAMPLE_SIZE is not None and cls.DEBUG_SAMPLE_SIZE < full_size:
            return cls.DEBUG_SAMPLE_SIZE
        return full_size
