import os
import torch

# ==========================================
# Global Configuration and Hyperparameters
# ==========================================

# Random Seed for Reproducibility
SEED = 42

# ------------------------------------------
# File Paths and Directories
# ------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_11"  # Cache directory for processed data
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary writeable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

# ------------------------------------------
# Data Processing Configuration
# ------------------------------------------
CACHE_DATA = True  # Set to True to enable caching mechanism in data loader
ATOM_MAP = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
NUM_ATOM_TYPES = 4

# ------------------------------------------
# Model Architecture Hyperparameters
# ------------------------------------------
# Atomic Stream (Wide Point Processor)
ATOMIC_INPUT_DIM = (
    4 + 3 + 3 + 1
)  # One-hot (4) + Centered XYZ (3) + Fractional (3) + PBC Neighbor Dist (1)
ATOMIC_HIDDEN_DIM = 512
ATOMIC_LAYERS = 3
ATOMIC_DROPOUT = 0.3

# Global Stream (Thermodynamic Context)
# Lattice lengths (3) + Angles (3) + Volume (1) + Density (1) + Stoichiometry (4)
GLOBAL_INPUT_DIM = 12
GLOBAL_HIDDEN_DIM = 256
GLOBAL_DROPOUT = 0.3

# Fusion Head
# Global Mean (512) + Global Max (512) + 4 Element Means (4*512) + Global Emb (256)
FUSION_INPUT_DIM = 512 + 512 + (4 * 512) + 256
FUSION_HIDDEN_DIM = 256
FUSION_DROPOUT = 0.2
OUTPUT_DIM = 2  # Formation energy, Bandgap energy

# ------------------------------------------
# Training Hyperparameters
# ------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 1e-3  # Initial learning rate
WEIGHT_DECAY = 1e-4  # Regularization for wide layers
EPOCHS = 200  # Total training epochs
PATIENCE = 20  # Early stopping patience

# Scheduler settings (ReduceLROnPlateau)
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
MIN_LR = 1e-6

# ------------------------------------------
# Computation Settings
# ------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of dataloader workers
