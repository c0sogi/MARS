import os
import torch


class Config:
    # ==========================================
    # System & Reproducibility
    # ==========================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # Adjust based on vCPUs (12 available)

    # ==========================================
    # File Paths
    # ==========================================
    # Root directories
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"

    # Specific sub-directory for this idea/experiment
    IDEA_WORK_DIR = os.path.join(WORKING_DIR, "idea_1")
    os.makedirs(IDEA_WORK_DIR, exist_ok=True)

    # Input Files
    STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Metadata Files (Pre-split)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output Paths
    MODEL_SAVE_PATH = os.path.join(IDEA_WORK_DIR, "best_model.pt")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # ==========================================
    # Data Processing Hyperparameters
    # ==========================================
    # Graph Construction
    CUTOFF_RADIUS = 3.0  # Angstroms: Maximum distance to define an edge between atoms

    # Mappings
    # Atoms found in QM9/CHAMPS: H, C, N, O, F
    ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}

    # Scalar Coupling Types
    TYPE_MAP = {
        "1JHC": 0,
        "2JHC": 1,
        "3JHC": 2,
        "1JHN": 3,
        "2JHN": 4,
        "3JHN": 5,
        "2JHH": 6,
        "3JHH": 7,
    }

    # Inverse mappings for analysis if needed
    INV_ATOM_MAP = {v: k for k, v in ATOM_MAP.items()}
    INV_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}

    # Normalization (Approximate stats from EDA)
    TARGET_MEAN = 15.9
    TARGET_STD = 34.9

    # Debugging
    # Set to a small integer (e.g., 5000) to limit dataset size during development
    # Set to None to use full dataset
    DEBUG_SAMPLE_SIZE = None

    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Embedding Dimensions
    ATOM_EMBED_DIM = 64
    TYPE_EMBED_DIM = 32

    # GCN Architecture
    HIDDEN_DIM = 128
    NUM_GCN_LAYERS = 3
    DROPOUT = 0.1

    # Readout/MLP Architecture
    MLP_HIDDEN_DIM = 128

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 30

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 5

    # Scheduler
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 2
