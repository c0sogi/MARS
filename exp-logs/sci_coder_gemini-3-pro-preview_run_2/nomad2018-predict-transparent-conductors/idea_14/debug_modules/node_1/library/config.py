import os
import torch
import random
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # Directories and Paths
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"

    # Sub-directories for artifacts
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"

    # Metadata Files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Paths (templates)
    TRAIN_GEOMETRY_DIR = os.path.join(INPUT_DIR, "train")
    TEST_GEOMETRY_DIR = os.path.join(INPUT_DIR, "test")

    # Cache Files
    TRAIN_GRAPHS_CACHE = os.path.join(CACHE_DIR, "train_graphs.npz")
    VAL_GRAPHS_CACHE = os.path.join(CACHE_DIR, "val_graphs.npz")
    TEST_GRAPHS_CACHE = os.path.join(CACHE_DIR, "test_graphs.npz")
    TARGET_SCALER_PATH = os.path.join(CACHE_DIR, "target_scaler.npz")

    # Model Checkpoint
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    # Submission
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
    SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

    # -------------------------------------------------------------------------
    # Data Processing Hyperparameters
    # -------------------------------------------------------------------------
    # Graph Construction
    CUTOFF_RADIUS = 5.0  # Angstroms, strict cutoff for local environment
    MAX_NEIGHBORS = 12  # Maximum number of neighbors per atom to consider

    # RBF Expansion
    RBF_MIN = 0.0
    RBF_MAX = 5.0
    NUM_RBF_BINS = 60  # Number of Gaussian basis functions

    # Targets
    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # -------------------------------------------------------------------------
    # Model Architecture Hyperparameters (CGCNN-IB)
    # -------------------------------------------------------------------------
    ATOM_EMBEDDING_DIM = 128
    EDGE_EMBEDDING_DIM = 128  # Projected from RBF features

    # Interaction Blocks
    NUM_INTERACTION_BLOCKS = 4

    # Inverted Bottleneck FFN
    FFN_HIDDEN_DIM = 256  # Expansion dimension inside the block

    # Regularization
    DROPOUT_RATE = 0.1

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    RANDOM_SEED = 42
    BATCH_SIZE = 48
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 150
    EARLY_STOPPING_PATIENCE = 20

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For DataLoader

    @staticmethod
    def setup():
        """
        Ensures all necessary directories exist and sets random seeds.
        """
        # Create directories
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Set seeds
        random.seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        torch.manual_seed(Config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.RANDOM_SEED)
            # Ensure deterministic behavior for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Initialize environment immediately upon import
Config.setup()
