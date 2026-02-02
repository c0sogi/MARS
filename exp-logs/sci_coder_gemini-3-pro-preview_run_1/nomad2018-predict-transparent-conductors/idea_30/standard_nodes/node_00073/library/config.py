import os

# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_30"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache File Paths (for deterministic data processing)
TRAIN_CACHE_PATH = os.path.join(WORKING_DIR, "train_data.npz")
VAL_CACHE_PATH = os.path.join(WORKING_DIR, "val_data.npz")
TEST_CACHE_PATH = os.path.join(WORKING_DIR, "test_data.npz")
SCALERS_CACHE_PATH = os.path.join(WORKING_DIR, "scalers.npz")

# Model Save Path
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pt")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# PHYSICAL CONSTANTS & ATOMIC PROPERTIES
# =============================================================================
# Used for physics-injection in the Global Stream.
# Values: Atomic Mass (u), Covalent Radius (Angstroms), Pauling Electronegativity
ATOMIC_PROPERTIES = {
    "Al": {"mass": 26.981539, "radius": 1.21, "electronegativity": 1.61},
    "Ga": {"mass": 69.723, "radius": 1.22, "electronegativity": 1.81},
    "In": {"mass": 114.818, "radius": 1.42, "electronegativity": 1.78},
    "O": {"mass": 15.999, "radius": 0.66, "electronegativity": 3.44},
}

# Mapping for One-Hot Encoding of Atomic Identity
ATOM_TO_INDEX = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
NUM_ATOM_TYPES = len(ATOM_TO_INDEX)

# =============================================================================
# DATA PROCESSING HYPERPARAMETERS
# =============================================================================
# Neighbors for Local Packing Density calculation
K_NEIGHBORS = 12

# Atomic Feature Dimensions
# 4 (One-Hot) + 3 (Coords x,y,z) + 1 (NN Dist) + 1 (Local Packing Density)
ATOMIC_INPUT_DIM = 9

# Global Feature Dimensions
# 3 (Lattice Vecs) + 3 (Angles) + 1 (Volume) + 1 (Density) + 1 (Total Atoms) +
# 3 (Stoichiometry) + 3 (Weighted Mass, Radius, Electronegativity)
GLOBAL_INPUT_DIM = 15

# =============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# =============================================================================
# Atomic Stream (Wide Point Processor)
ATOMIC_HIDDEN_DIM = 512
ATOMIC_LAYERS = 3
ATOMIC_DROPOUT = 0.1
ATOMIC_EMBEDDING_DIM = 128  # Dimension after aggregation

# Global Stream (Physics-Enhanced Context)
GLOBAL_HIDDEN_DIM = 256
GLOBAL_LAYERS = 2
GLOBAL_DROPOUT = 0.1
GLOBAL_EMBEDDING_DIM = 64

# Fusion Head
# Concatenation size: (ATOMIC_EMBEDDING_DIM * 2 for mean+max pool) + GLOBAL_EMBEDDING_DIM
# (128 * 2) + 64 = 320
FUSION_HIDDEN_DIMS = [256, 128]
FUSION_DROPOUT = 0.1
OUTPUT_DIM = 2  # formation_energy_ev_natom, bandgap_energy_ev

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 20  # Early stopping patience
FACTOR = 0.5  # LR Scheduler factor
MIN_LR = 1e-6  # LR Scheduler minimum learning rate
