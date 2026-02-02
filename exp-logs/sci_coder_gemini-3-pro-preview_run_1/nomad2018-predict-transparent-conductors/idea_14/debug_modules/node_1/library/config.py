import os
import torch
import random
import numpy as np


def set_seed(seed):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Config:
    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_14"
    SUBMISSION_DIR = "./submission"

    # Ensure working and submission directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # File Paths
    # -------------------------------------------------------------------------
    # Metadata CSVs
    TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
    VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
    TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

    # Sample Submission
    SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

    # Cache Files (using .pt for PyTorch tensors/objects)
    TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.pt")
    VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.pt")
    TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.pt")
    SCALER_CACHE_PATH = os.path.join(WORKING_DIR, "scalers.pt")

    # Model Checkpoint
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # -------------------------------------------------------------------------
    # Data Configuration
    # -------------------------------------------------------------------------
    ATOM_TYPES = ["Al", "Ga", "In", "O"]
    ATOM_TO_IDX = {atom: i for i, atom in enumerate(ATOM_TYPES)}
    NUM_ATOM_TYPES = len(ATOM_TYPES)

    TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

    # Debugging: Set to a small integer (e.g., 100) to limit dataset size
    # Set to None for full training
    DEBUG_SAMPLE_SIZE = None

    # -------------------------------------------------------------------------
    # Model Hyperparameters
    # -------------------------------------------------------------------------
    # Atomic Stream
    ATOMIC_HIDDEN_DIM = 512  # Wide MLP
    ATOMIC_LATENT_DIM = 128

    # Global Stream
    GLOBAL_HIDDEN_DIM = 256
    GLOBAL_LATENT_DIM = 64

    # Fusion Head
    FUSION_HIDDEN_DIM = 256

    # Regularization
    DROPOUT_RATE = 0.2
    USE_BATCH_NORM = True

    # -------------------------------------------------------------------------
    # Training Hyperparameters
    # -------------------------------------------------------------------------
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 200

    # Early Stopping
    PATIENCE = 20

    # Scheduler (ReduceLROnPlateau)
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 10
    SCHEDULER_MIN_LR = 1e-6

    # -------------------------------------------------------------------------
    # Hardware
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # For data loading
