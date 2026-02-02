import os
import torch

# ==========================================
# Global System Settings
# ==========================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # Optimized for the 12 vCPU environment

# ==========================================
# File Paths & Directories
# ==========================================
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_15"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Files
STRUCTURES_PATH = os.path.join(INPUT_DIR, "structures.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Pre-split by molecule to prevent leakage)
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Files
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# Cache Files (Parquet/NPY for fast loading)
# We use specific names for the molecule-parallel flattened format
TRAIN_CACHE_DIR = os.path.join(WORKING_DIR, "train_cache")
VAL_CACHE_DIR = os.path.join(WORKING_DIR, "val_cache")
TEST_CACHE_DIR = os.path.join(WORKING_DIR, "test_cache")

# ==========================================
# Data Processing & Physical Constants
# ==========================================
# Atom Type Mapping (Canonical)
# Based on dataset analysis: H, C, N, O, F are the only elements present
ATOM_MAP = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
NUM_ATOM_TYPES = len(ATOM_MAP)

# Coupling Type Mapping (Canonical)
# Used for embedding the coupling type in the conditional readout
TYPE_MAP = {
    "1JHC": 0,
    "2JHH": 1,
    "1JHN": 2,
    "2JHN": 3,
    "2JHC": 4,
    "3JHH": 5,
    "3JHC": 6,
    "3JHN": 7,
}
INV_TYPE_MAP = {v: k for k, v in TYPE_MAP.items()}
NUM_COUPLING_TYPES = len(TYPE_MAP)

# Normalization Constants (Will be computed/loaded during runtime)
# Placeholder for standardization stats file
STATS_PATH = os.path.join(WORKING_DIR, "stats.npy")

# ==========================================
# Model Hyperparameters (MP-IN Architecture)
# ==========================================
# Backbone: Continuous Filter Convolution (SchNet-like)
HIDDEN_DIM = 128  # Dimension of node and edge embeddings
NUM_INTERACTIONS = 6  # Number of interaction blocks (depth)
RBF_RADIUS = 5.0  # Cutoff radius in Angstroms for graph construction
NUM_RBF = 50  # Number of Gaussian Radial Basis Functions
ACTIVATION = "silu"  # Activation function (SiLU/Swish)

# Readout: Interaction-Aware Conditional Head
READOUT_HIDDEN_DIM = 128  # Hidden dimension for the MLP head
DROPOUT = 0.0  # Dropout rate (usually 0 for regression on physics data)

# ==========================================
# Training Hyperparameters
# ==========================================
# Optimization
BATCH_SIZE = 96  # Number of MOLECULES per batch (not pairs)
LEARNING_RATE = 5e-4  # Initial learning rate
WEIGHT_DECAY = 1e-8  # L2 regularization
GRAD_CLIP = 1.0  # Gradient clipping norm

# Scheduler (Cosine Annealing Warm Restarts)
SCHEDULER_T_0 = 10  # Number of epochs for the first restart
SCHEDULER_T_MULT = 2  # Factor to increase restart interval
MIN_LR = 1e-6  # Minimum learning rate

# Loop Control
MAX_EPOCHS = 50  # Maximum training epochs
PATIENCE = 7  # Early stopping patience
EARLY_STOP_METRIC = "val_mae"  # Metric to monitor

# Debugging / Development
# Set DEBUG = True to run on a tiny subset of data for pipeline verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 1000  # Number of molecules to use if DEBUG is True
