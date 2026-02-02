import os

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
WORKING_DIR = os.path.join(BASE_DIR, "working", "idea_42")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw data files
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache files
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npz")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.npz")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npz")
SCALERS_PATH = os.path.join(WORKING_DIR, "scalers.npz")

# Model checkpoint
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
SEED = 42

# =============================================================================
# FEATURE EXTRACTION HYPERPARAMETERS
# =============================================================================
# Neighbor counts for multi-scale context and ratio calculation
K_MIN = 1  # For d_min
K_SHORT = 6  # Short-range chemical context
K_MEDIUM = 12  # Medium-range (used for Packing Ratio denominator)
K_LONG = 24  # Long-range chemical context

# Atomic species mapping
ATOM_MAP = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
NUM_ATOM_TYPES = len(ATOM_MAP)

# Feature Dimensions
# Atomic Stream:
#   - Identity (One-hot): 4
#   - Centered Coords (x,y,z): 3
#   - d_min: 1
#   - Packing Ratio: 1
#   - Context K=6 (Weighted Composition): 4
#   - Context K=24 (Weighted Composition): 4
ATOM_FEATURES_DIM = 4 + 3 + 1 + 1 + 4 + 4  # Total: 17

# Global Stream:
#   - Lattice Lengths (a, b, c): 3
#   - Lattice Angles (alpha, beta, gamma): 3
#   - Volume: 1
#   - Density: 1
#   - Stoichiometry (Al, Ga, In): 3
#   - Total Atoms: 1
#   - Lattice Aspect Ratios (a/b, b/c, c/a): 3
GLOBAL_FEATURES_DIM = 3 + 3 + 1 + 1 + 3 + 1 + 3  # Total: 15

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# Wide MLP settings
HIDDEN_DIM = 512
ATOMIC_LAYERS = 3
GLOBAL_LAYERS = 2
FUSION_LAYERS = 3
DROPOUT = 0.1
USE_BATCH_NORM = True

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 20  # Early stopping patience
FACTOR = 0.5  # ReduceLROnPlateau factor
MIN_LR = 1e-6  # ReduceLROnPlateau min_lr

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# Set to None to use full dataset, or an integer for debugging with a subset
DEBUG_SAMPLE_SIZE = None
