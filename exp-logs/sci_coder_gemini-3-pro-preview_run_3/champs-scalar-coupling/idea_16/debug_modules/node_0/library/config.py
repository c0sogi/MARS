import os
import torch
import numpy as np
import random


class Config:
    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_16"
    SUBMISSION_DIR = "./submission"

    # Input Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")

    # Metadata Files (Pre-split)
    TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Cache Files (for flattened SoA data)
    # We use a directory structure for caching processed numpy arrays
    TRAIN_CACHE_DIR = os.path.join(WORKING_DIR, "train_cache")
    VAL_CACHE_DIR = os.path.join(WORKING_DIR, "val_cache")
    TEST_CACHE_DIR = os.path.join(WORKING_DIR, "test_cache")

    # Statistics for Target Standardization
    STATS_PATH = os.path.join(WORKING_DIR, "stats.npy")

    # Model Checkpoints & Outputs
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data Settings & Mappings
    # ==========================================
    # Atom Type Mapping (H, C, N, O, F are the elements in this dataset)
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # Coupling Type Mapping
    COUPLING_TYPE_MAP = {
        "1JHC": 0,
        "2JHC": 1,
        "3JHC": 2,
        "1JHN": 3,
        "2JHN": 4,
        "3JHN": 5,
        "2JHH": 6,
        "3JHH": 7,
    }
    INVERSE_COUPLING_TYPE_MAP = {v: k for k, v in COUPLING_TYPE_MAP.items()}
    NUM_COUPLING_TYPES = len(COUPLING_TYPE_MAP)

    # ==========================================
    # 3. Model Hyperparameters (SDIN)
    # ==========================================
    # General
    HIDDEN_DIM = 128  # Dimension of node embeddings and interaction features
    N_LAYERS = 6  # Number of continuous filter interaction blocks

    # RBF / Geometry
    N_RBF = 25  # Number of Gaussian Radial Basis Functions
    CUTOFF = 5.0  # Interaction radius in Angstroms

    # Readout Head
    COUPLING_EMBED_DIM = 16  # Dimension for coupling type embedding
    USE_COUPLING_EMBED = True  # Whether to use the type embedding

    # ==========================================
    # 4. Training Configuration
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    BATCH_SIZE = 48  # Number of MOLECULES per batch (each contains multiple couplings)
    LEARNING_RATE = 5e-4  # Initial learning rate
    WEIGHT_DECAY = 1e-5  # L2 regularization
    MAX_GRAD_NORM = 1.0  # Gradient clipping

    # Scheduler (Cosine Annealing Warm Restarts)
    T_0 = 10  # Number of epochs for the first restart
    T_MULT = 2  # Factor to increase restart period
    ETA_MIN = 1e-6  # Minimum learning rate

    # Training Loop
    MAX_EPOCHS = 30  # Maximum number of training epochs
    PATIENCE = 5  # Early stopping patience
    NUM_WORKERS = 4  # DataLoader workers

    # Debugging / Development
    DEBUG = False  # If True, runs on a small subset of data
    DEBUG_SAMPLE_SIZE = 1000  # Number of molecules to use in debug mode

    @classmethod
    def setup(cls):
        """Ensures necessary directories exist and sets random seeds."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)
        os.makedirs(cls.TRAIN_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.VAL_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.TEST_CACHE_DIR, exist_ok=True)

        # Set random seeds for reproducibility
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cls.SEED)
            torch.cuda.manual_seed_all(cls.SEED)
            # Ensure deterministic behavior for cuDNN
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Initialize setup immediately when module is imported to guarantee directories and seeds
Config.setup()
