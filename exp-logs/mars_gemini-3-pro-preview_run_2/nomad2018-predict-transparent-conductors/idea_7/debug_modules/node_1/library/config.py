import os

# =============================================================================
# File Paths and Directories
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_7"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission File
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache Files
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_graphs.npz")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_graphs.npz")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_graphs.npz")
SCALER_CACHE_PATH = os.path.join(WORKING_DIR, "scalers.npy")

# Model Checkpoint
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# =============================================================================
# Data Parameters
# =============================================================================
# Atoms present in the dataset
ATOM_TYPES = ["Al", "Ga", "In", "O"]
ATOM_MAP = {atom: i for i, atom in enumerate(ATOM_TYPES)}
NUM_ATOM_TYPES = len(ATOM_TYPES)

# Graph Construction
CUTOFF_RADIUS = 5.0  # Angstroms
MAX_NEIGHBORS = 12  # Maximum number of neighbors per node

# Global Features
# 6 Lattice params (a, b, c, alpha, beta, gamma) + 4 Composition fractions
NUM_GLOBAL_FEATURES = 10

# =============================================================================
# Model Hyperparameters
# =============================================================================
# General
SEED = 42

# Architecture
HIDDEN_DIM = 64
N_GCN_LAYERS = 4
RBF_BINS = 60
RBF_GAMMA = 0.5  # Width parameter for RBF
DROPOUT = 0.1

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 48
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15

# Targets
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
NUM_TARGETS = len(TARGET_COLS)
