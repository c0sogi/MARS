import os

# ==========================================
# Atomic Properties
# ==========================================
# Mapping: Symbol -> Property
# Used for Physics-Aware Feature Injection

# Atomic Mass (u)
ATOMIC_MASSES = {
    "Al": 26.981539,
    "Ga": 69.723,
    "In": 114.818,
    "O": 15.999,
}

# Covalent Radius (Angstroms) - roughly based on Pyykko
COVALENT_RADII = {
    "Al": 1.21,
    "Ga": 1.22,
    "In": 1.42,
    "O": 0.66,
}

# Pauling Electronegativity
ELECTRONEGATIVITY = {
    "Al": 1.61,
    "Ga": 1.81,
    "In": 1.78,
    "O": 3.44,
}

# Atomic Number mapping for one-hot encoding ordering
# Order: Al, Ga, In, O
ATOMIC_LABELS = ["Al", "Ga", "In", "O"]
ATOM_TO_INDEX = {atom: i for i, atom in enumerate(ATOMIC_LABELS)}

# ==========================================
# Data Configuration
# ==========================================

# Directory Paths
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_48"  # Cache directory for this specific idea
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache File Names (Parquet or NPZ)
CACHE_TRAIN_DATA = os.path.join(WORKING_DIR, "train_data.npz")
CACHE_VAL_DATA = os.path.join(WORKING_DIR, "val_data.npz")
CACHE_TEST_DATA = os.path.join(WORKING_DIR, "test_data.npz")
CACHE_SCALERS = os.path.join(WORKING_DIR, "scalers.npz")

# Feature Dimensions
# Atomic Stream Features:
# 4 (One-hot) + 3 (Coords) + 1 (d_min) + 1 (Packing) + 8 (Context K=6,24) + 4 (Proximity) = 21
ATOMIC_FEATURE_DIM = 21

# Global Stream Features:
# 3 (Lattice Lens) + 3 (Angles) + 1 (Vol) + 3 (Aspect Ratios) +
# 1 (Density) + 1 (Total Atoms) + 3 (Stoich) +
# 3 (Phys Means) + 3 (Phys Vars) = 21
GLOBAL_FEATURE_DIM = 21

# Target Columns
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
NUM_TARGETS = len(TARGET_COLS)

# Neighbor Search Parameters
NEIGHBOR_K_SHORT = 6
NEIGHBOR_K_LONG = 24

# ==========================================
# Model Hyperparameters
# ==========================================

# Training
BATCH_SIZE = 32  # Smaller batch size for sparse batching stability
LEARNING_RATE = 5e-4  # Moderate learning rate
WEIGHT_DECAY = 1e-4  # Regularization
NUM_EPOCHS = 200  # Maximum training epochs
EARLY_STOPPING_PATIENCE = 20

# Architecture
HIDDEN_DIM = 512  # Wide MLP layers
LATENT_DIM = 256  # Dimension after pooling/fusion
DROPOUT_RATE = 0.1  # Dropout probability
USE_BATCH_NORM = True  # Enable Batch Normalization

# Random Seed
SEED = 42
