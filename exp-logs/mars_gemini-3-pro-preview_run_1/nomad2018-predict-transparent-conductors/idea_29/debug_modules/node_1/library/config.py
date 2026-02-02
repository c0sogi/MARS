import os

# =============================================================================
# Directories and Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_29"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# File Paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# Physics / Chemical Constants
# =============================================================================
# Atomic Properties: {Symbol: [Mass (u), Covalent Radius (Angstrom), Electronegativity (Pauling)]}
# These values are used to compute composition-weighted global features.
ATOMIC_PROPERTIES = {
    "Al": [26.981539, 1.21, 1.61],
    "Ga": [69.723, 1.22, 1.81],
    "In": [114.818, 1.42, 1.78],
    "O": [15.999, 0.66, 3.44],
}

# Mapping for one-hot encoding of atomic identity in the local stream
ATOM_TO_INDEX = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
NUM_ATOM_TYPES = len(ATOM_TO_INDEX)

# =============================================================================
# Model Hyperparameters
# =============================================================================
# Atomic Stream (Wide MLP)
ATOMIC_HIDDEN_DIM = 512
ATOMIC_LAYERS = 3

# Global Stream (High Capacity)
GLOBAL_HIDDEN_DIM = 256
GLOBAL_LAYERS = 2

# Fusion Head
FUSION_HIDDEN_DIM = 256

# Regularization
DROPOUT = 0.1

# =============================================================================
# Training Configuration
# =============================================================================
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 200
PATIENCE = 20  # Early stopping patience

# =============================================================================
# Data Processing Configuration
# =============================================================================
# Features to exclude from global tabular input (since they are handled specifically or are IDs)
EXCLUDE_COLS = ["id", "file_path", "formation_energy_ev_natom", "bandgap_energy_ev"]

# Target columns
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]
