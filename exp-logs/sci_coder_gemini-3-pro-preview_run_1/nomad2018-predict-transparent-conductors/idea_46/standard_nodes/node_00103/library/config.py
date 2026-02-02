import os

# ==========================================
# Paths and Directories
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_46"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Geometry directory (relative to input)
GEOMETRY_DIR = INPUT_DIR

# Cache paths for processed data
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npz")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.npz")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npz")
SCALERS_CACHE_PATH = os.path.join(WORKING_DIR, "scalers.npz")

# Model save path
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
SUBMISSION_PATH = "./submission/submission.csv"

# ==========================================
# Physical Constants & Lookups
# ==========================================
# Covalent Radii (Angstroms) - Used for Bond Hardness Proxy
# Values approx from PyMatGen or standard tables
COVALENT_RADII = {"Al": 1.21, "Ga": 1.22, "In": 1.42, "O": 0.66}

# Atomic Masses (amu) - Used for Weighted Physics
ATOMIC_MASSES = {"Al": 26.9815, "Ga": 69.723, "In": 114.818, "O": 15.999}

# Pauling Electronegativity - Used for Weighted Physics
ELECTRONEGATIVITY = {"Al": 1.61, "Ga": 1.81, "In": 1.78, "O": 3.44}

# Atom type mapping for one-hot encoding
ATOM_TYPES = ["Al", "Ga", "In", "O"]
ATOM_TO_IDX = {atom: i for i, atom in enumerate(ATOM_TYPES)}

# ==========================================
# Feature Configuration
# ==========================================
# Neighbor counts for multi-scale chemical context
K_NEIGHBORS_LIST = [6, 24]

# Input Dimensions
# Atomic Features:
# 4 (One-hot) + 3 (Coords) + 1 (d_min) + 1 (Packing) + 1 (Bond Hardness) +
# 4 (Context K=6) + 4 (Context K=24) = 18
ATOM_INPUT_DIM = 18

# Global Features:
# 3 (Lattice Lens) + 3 (Lattice Angs) + 1 (Vol) + 1 (Density) + 4 (Stoich) +
# 1 (Total Atoms) + 3 (Aspect Ratios) + 3 (Weighted Physics) + 1 (Ang Distortion) = 20
GLOBAL_INPUT_DIM = 20

# ==========================================
# Model Hyperparameters
# ==========================================
ATOM_HIDDEN_DIM = 512  # Width of atomic encoder MLP
GLOBAL_HIDDEN_DIM = 256  # Width of global encoder MLP
DROPOUT_RATE = 0.1  # Dropout probability
GATE_DIM = ATOM_HIDDEN_DIM * 2  # Dimension after dual pooling (Mean + Max)

# ==========================================
# Training Hyperparameters
# ==========================================
SEED = 42
BATCH_SIZE = 64  # Number of crystals per batch
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 20

# Debugging flag to use a smaller subset of data
DEBUG_MODE = False
DEBUG_SUBSET_SIZE = 100
