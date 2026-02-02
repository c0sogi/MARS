import os
import torch


class Config:
    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    SEED = 42

    # -------------------------------------------------------------------------
    # File Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Working Directory (Idea 22)
    WORKING_DIR = "./working/idea_22"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    CUTOFF_RADIUS = 5.0  # Angstroms (Strictly set as per description)
    MAX_NEIGHBORS = 50  # Maximum number of neighbors to consider per atom

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters (RA-CGN-AR)
    # -------------------------------------------------------------------------
    ATOM_EMBEDDING_DIM = 160  # Increased capacity (Cite solution_lesson_node_00005)
    EDGE_EMBEDDING_DIM = 160  # Increased capacity (Cite solution_lesson_node_00005)
    RBF_BINS = 60
    RBF_LOWER = 0.0
    RBF_UPPER = 5.0
    NUM_LAYERS = 5  # Increased depth (Cite solution_lesson_node_00005)
    DROPOUT_RATE = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 48
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 150
    PATIENCE = 15  # For early stopping

    # -------------------------------------------------------------------------
    # Target Variables
    # -------------------------------------------------------------------------
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # -------------------------------------------------------------------------
    # Compute
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    # -------------------------------------------------------------------------
    # Debugging
    # -------------------------------------------------------------------------
    DEBUG = False
    SUBSET_SIZE = None  # Set to an integer to train on a subset
