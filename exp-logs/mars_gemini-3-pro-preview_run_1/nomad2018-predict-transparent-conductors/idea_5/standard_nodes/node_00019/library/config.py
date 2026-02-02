import os
import torch

# ==========================================
#              Global Configuration
# ==========================================

# Random Seed for reproducibility
SEED = 42

# ==========================================
#              File Paths
# ==========================================

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"
SUBMISSION_DIR = "./submission"

# Input Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Files
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")

# Cache Files
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npz")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.npz")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npz")

# ==========================================
#           Data Processing Config
# ==========================================

# Atom definitions
ATOM_TYPES = ["Al", "Ga", "In", "O"]
NUM_ATOM_TYPES = len(ATOM_TYPES)

# Topological Fingerprinting
K_NEIGHBORS = 16  # Number of nearest neighbors to include in local fingerprint

# Global Features
# 3 Lattice Lengths + 3 Lattice Angles + 3 Compositions (Al, Ga, In) + 1 Volume + 1 Density
NUM_GLOBAL_FEATURES = 11

# ==========================================
#           Model Architecture
# ==========================================

# Global Context Stream
GLOBAL_HIDDEN_DIM = 128
GLOBAL_CONTEXT_DIM = 64  # Size of the vector injected into the atomic stream

# Atomic Stream
# Base atomic features: 4 (One-Hot) + 3 (Coords) + K_NEIGHBORS (Distances)
BASE_ATOMIC_DIM = NUM_ATOM_TYPES + 3 + K_NEIGHBORS
# Total input to atomic encoder includes the injected global context
ATOMIC_INPUT_DIM = BASE_ATOMIC_DIM + GLOBAL_CONTEXT_DIM

ATOMIC_HIDDEN_DIM = 512  # Wide MLP for atomic processing
LATENT_DIM = 512  # Dimension of atomic embeddings before pooling

# Prediction Head
HEAD_HIDDEN_DIM = 256
DROPOUT_RATE = 0.1

# ==========================================
#           Training Hyperparameters
# ==========================================

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15

# ==========================================
#           System & Debugging
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# Set to an integer (e.g., 100) to limit dataset size for rapid debugging, or None for full run
MAX_SAMPLES = None


def setup_directories():
    """
    Ensures that the necessary working and submission directories exist.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
