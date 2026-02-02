import os
import torch

# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================
# Base paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_25"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache file paths for processed data
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.pt")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.pt")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.pt")
SCALERS_CACHE_PATH = os.path.join(WORKING_DIR, "scalers.pt")

# Model checkpoint path
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA PROCESSING HYPERPARAMETERS
# =============================================================================
# Random seed for reproducibility
SEED = 42

# Debugging: Set to a small integer (e.g., 100) to limit dataset size for testing.
# Set to None to use the full dataset.
DEBUG_SAMPLE_SIZE = None

# Feature Engineering Constants
LARGE_DISTANCE_CONSTANT = 100.0  # Value for missing hetero-atomic neighbors
ATOM_TYPES = ["Al", "Ga", "In", "O"]
NUM_ATOM_TYPES = len(ATOM_TYPES)

# Input Dimensions
# Atomic Stream: One-hot (4) + Coords (3) + Homo-dist (1) + Hetero-dist (1)
ATOMIC_INPUT_DIM = 9
# Global Stream: Lattice lengths (3) + Angles (3) + Volume (1) + Density (1) + Total Atoms (1) + Composition (3)
GLOBAL_INPUT_DIM = 12

# =============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# =============================================================================
# Atomic Stream Encoder (Wide MLP)
ATOMIC_HIDDEN_DIM = 512
ATOMIC_LAYERS = 3

# Global Stream Encoder
GLOBAL_HIDDEN_DIM = 256
GLOBAL_LAYERS = 2

# Fusion Head
FUSION_HIDDEN_DIM = 256
FUSION_LAYERS = 3

# Regularization
DROPOUT_RATE = 0.3
USE_BATCH_NORM = True

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # Regularization for wide layers
EPOCHS = 200
PATIENCE = 20  # Early stopping patience
NUM_WORKERS = 4  # For data loading

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# TARGET CONFIGURATION
# =============================================================================
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
NUM_TARGETS = len(TARGET_COLS)
