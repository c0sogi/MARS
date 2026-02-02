import os
import torch

# -----------------------------------------------------------------------------
# General Configuration
# -----------------------------------------------------------------------------
SEED = 42
DEBUG = False  # Set to True to run on a small subset of data for debugging
DEBUG_SAMPLE_SIZE = 2000  # Number of molecules/samples to use in debug mode

# Compute Resources
# We use the available NVIDIA A100 GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12  # Utilizing the 12 vCPUs for data loading
PIN_MEMORY = True

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
WORKING_DIR = os.path.join(BASE_DIR, "working")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Specific Data Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Structure Files
# The idea specifies using raw XYZ files for inference/graph construction
STRUCTURES_DIR = os.path.join(INPUT_DIR, "structures")
STRUCTURES_CSV = os.path.join(INPUT_DIR, "structures.csv")

# Submission
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Caching
# Directory to store pre-processed graphs and basis function expansions
IDEA_NAME = "idea_3"
CACHE_DIR = os.path.join(WORKING_DIR, IDEA_NAME)

# Create necessary directories
os.makedirs(SUBMISSION_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Data Processing Constants
# -----------------------------------------------------------------------------
# Atom types found in the dataset
ATOM_TYPES = ["H", "C", "N", "O", "F"]
ATOM_TO_INT = {atom: i for i, atom in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)

# Scalar Coupling Types
COUPLING_TYPES = ["1JHC", "2JHC", "3JHC", "1JHN", "2JHN", "3JHN", "2JHH", "3JHH"]
COUPLING_TO_INT = {ctype: i for i, ctype in enumerate(COUPLING_TYPES)}
NUM_COUPLING_TYPES = len(COUPLING_TYPES)

# Geometric Graph Construction
CUTOFF = 5.0  # Angstroms, spatial cutoff for neighbor detection

# -----------------------------------------------------------------------------
# Model Hyperparameters (DMPNN + Spherical Basis)
# -----------------------------------------------------------------------------
# Architecture Dimensions
HIDDEN_DIM = 192  # Dimension of atom and edge embeddings
NUM_INTERACTIONS = 6  # Number of message passing interaction blocks
OUTPUT_DIM = 1  # Regression target (scalar coupling constant)

# Basis Function Expansions
# Radial Basis Functions (RBF) for distances
NUM_RBF = 128
# Spherical Basis Functions (SBF) for angles/triplets
NUM_SPHERICAL = 7  # L_max (angular resolution)
NUM_RADIAL = 6  # N_max (radial resolution within spherical basis)
ENVELOPE_EXPONENT = 5  # Polynomial envelope for smooth cutoff

# Regularization & Optimization
# Dropout is explicitly avoided based on the idea description (Deterministic Readout)
USE_DROPOUT = False

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 64  # Batch size for graph data
LEARNING_RATE = 1e-4  # Initial learning rate
WEIGHT_DECAY = 1e-5  # L2 Regularization
MAX_EPOCHS = 45  # Total training epochs
WARMUP_EPOCHS = 2  # Linear warmup period

# Scheduler
LR_PATIENCE = 5  # Reduce LR after n epochs of no improvement
LR_FACTOR = 0.5  # Factor to reduce LR
MIN_LR = 1e-6

# Target Normalization
# Normalize targets to mean=0, std=1 per coupling type
NORMALIZE_TARGETS = True


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_debug_mode(mode: bool = True):
    """
    Adjusts configuration for debugging purposes.
    Reduces epochs and batch size, and enables the DEBUG flag.
    """
    global DEBUG, MAX_EPOCHS, BATCH_SIZE
    DEBUG = mode
    if DEBUG:
        print(f"[Config] Debug mode ENABLED. Using subset size: {DEBUG_SAMPLE_SIZE}")
        MAX_EPOCHS = 2
        BATCH_SIZE = 16


def get_config_dict():
    """Returns the current configuration as a dictionary."""
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}
