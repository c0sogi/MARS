import os
import torch


class Config:
    # ==========================================
    # 1. File Paths & Directories
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_18"
    SUBMISSION_DIR = "./submission"

    # Input Files
    STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Directories (Subdirectories in working dir)
    CACHE_DIR_TRAIN = os.path.join(WORKING_DIR, "train_cache")
    CACHE_DIR_VAL = os.path.join(WORKING_DIR, "val_cache")
    CACHE_DIR_TEST = os.path.join(WORKING_DIR, "test_cache")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==========================================
    # 2. Data Processing & Mappings
    # ==========================================
    # Atom Type Mapping (Canonical)
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    NUM_ATOM_TYPES = len(ATOM_MAP)

    # Coupling Type Mapping (Canonical)
    COUPLING_TYPE_MAP = {
        "1JHC": 0,
        "1JHN": 1,
        "2JHC": 2,
        "2JHH": 3,
        "2JHN": 4,
        "3JHC": 5,
        "3JHH": 6,
        "3JHN": 7,
    }
    INVERSE_COUPLING_TYPE_MAP = {v: k for k, v in COUPLING_TYPE_MAP.items()}
    NUM_COUPLING_TYPES = len(COUPLING_TYPE_MAP)

    # Normalization Statistics (Mean, Std) per Coupling Type
    # Derived from EDA to ensure consistent standardization/inverse-transform
    COUPLING_TYPE_STATS = {
        "1JHC": {"mean": 94.9502, "std": 18.2511},
        "1JHN": {"mean": 47.5233, "std": 10.8969},
        "2JHC": {"mean": -0.2778, "std": 4.5058},
        "2JHH": {"mean": -10.2810, "std": 3.9853},
        "2JHN": {"mean": 3.1184, "std": 3.6624},
        "3JHC": {"mean": 3.6901, "std": 3.0759},
        "3JHH": {"mean": 4.7700, "std": 3.7061},
        "3JHN": {"mean": 0.9903, "std": 1.3208},
    }

    # ==========================================
    # 3. Model Hyperparameters (MP-DIN)
    # ==========================================
    # Architecture
    HIDDEN_DIM = 128  # Dimension of node embeddings and interaction features
    NUM_INTERACTIONS = 6  # Number of continuous filter convolution layers
    RBF_START = 0.0  # RBF start distance
    RBF_END = 5.0  # RBF end distance (Cutoff)
    NUM_RBF = 50  # Number of Gaussian RBF basis functions

    # Readout Head
    USE_COUPLING_EMB = True  # Concatenate coupling type embedding
    COUPLING_EMB_DIM = 32  # Dimension of coupling type embedding

    # ==========================================
    # 4. Training Configuration
    # ==========================================
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    BATCH_SIZE = 48  # Number of molecules per batch (Molecule-Parallel)
    LEARNING_RATE = 5e-4  # Initial learning rate
    WEIGHT_DECAY = 1e-6  # L2 Regularization
    MAX_EPOCHS = 35  # Max training epochs
    PATIENCE = 5  # Early stopping patience

    # Scheduler
    SCHEDULER_T_0 = 10  # Cosine Annealing T_0
    SCHEDULER_T_MULT = 2  # Cosine Annealing T_mult
    SCHEDULER_ETA_MIN = 1e-6  # Min LR

    # ==========================================
    # 5. Debugging & Runtime Control
    # ==========================================
    DEBUG = False  # Set to True to run on a small subset
    DEBUG_SAMPLE_SIZE = 2000  # Number of molecules to use in debug mode
    NUM_WORKERS = 4  # Data loading workers

    @staticmethod
    def setup_directories():
        """Ensures all necessary directories exist."""
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR_TRAIN, exist_ok=True)
        os.makedirs(Config.CACHE_DIR_VAL, exist_ok=True)
        os.makedirs(Config.CACHE_DIR_TEST, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    @staticmethod
    def get_coupling_stats_tensor(device):
        """Returns Mean and Std tensors ordered by coupling type index."""
        means = torch.zeros(Config.NUM_COUPLING_TYPES, device=device)
        stds = torch.zeros(Config.NUM_COUPLING_TYPES, device=device)

        for type_name, stats in Config.COUPLING_TYPE_STATS.items():
            idx = Config.COUPLING_TYPE_MAP[type_name]
            means[idx] = stats["mean"]
            stds[idx] = stats["std"]

        return means, stds
