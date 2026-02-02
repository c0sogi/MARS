import os
import torch
import random
import numpy as np

# ==========================================
# Path Configurations
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_23"
SUBMISSION_DIR = "./submission"

TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache paths for processed data using .npz format
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npz")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.npz")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npz")
SCALERS_CACHE_PATH = os.path.join(WORKING_DIR, "scalers.npz")

MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 20
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
MIN_LR = 1e-6

# ==========================================
# Model Architecture Constants
# ==========================================
# Atomic Stream Input Features:
# 1. Self Identity (One-hot, 4 dims: Al, Ga, In, O)
# 2. Spatial Context (Centered Coords, 3 dims: x, y, z)
# 3. Nearest Neighbor Distance (Scalar, 1 dim)
# 4. Nearest Neighbor Identity (One-hot, 4 dims)
# Total: 4 + 3 + 1 + 4 = 12
ATOM_INPUT_DIM = 12
ATOMIC_HIDDEN_DIM = 512  # Wide MLP for point processing

# Global Stream Input Features:
# 1. Lattice lengths (3 dims)
# 2. Lattice angles (3 dims)
# 3. Volume (1 dim)
# 4. Atomic Density (1 dim)
# 5. Stoichiometry (3 dims: Al, Ga, In fractions)
# 6. Total Atoms (1 dim)
# Total: 3 + 3 + 1 + 1 + 3 + 1 = 12
GLOBAL_INPUT_DIM = 12
GLOBAL_HIDDEN_DIM = 256

DROPOUT = 0.2

# ==========================================
# Device Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Utility Functions
# ==========================================
def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_directories():
    """Ensures necessary directories exist."""
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
