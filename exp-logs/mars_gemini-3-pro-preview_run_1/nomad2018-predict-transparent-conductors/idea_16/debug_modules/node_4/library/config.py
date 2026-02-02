import os

# =============================================================================
# PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_16"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")
SAMPLE_SUBMISSION_FILE = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache files
TRAIN_CACHE = os.path.join(WORKING_DIR, "train_data.npz")
VAL_CACHE = os.path.join(WORKING_DIR, "val_data.npz")
TEST_CACHE = os.path.join(WORKING_DIR, "test_data.npz")
SCALERS_CACHE = os.path.join(WORKING_DIR, "scalers.npz")
MODEL_CHECKPOINT = os.path.join(WORKING_DIR, "best_model.pt")

# =============================================================================
# DATA CONSTANTS
# =============================================================================
SEED = 42
ATOM_MAP = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
NUM_ATOM_TYPES = len(ATOM_MAP)
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# Feature Dimensions
# Atomic Stream: One-hot (4) + Centered Cartesian (3) + Centered Fractional (3) + NN Dist (1) + Potential (1)
ATOMIC_INPUT_DIM = 12

# Global Stream: Lattice Lengths (3) + Angles (3) + Volume (1) + Density (1) + Stoichiometry (3) + Total Atoms (1)
GLOBAL_INPUT_DIM = 12

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 200
PATIENCE = 20  # Early stopping patience
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.3

# Scheduler settings
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6

# Model Architecture
ATOMIC_HIDDEN_DIM = 512
GLOBAL_HIDDEN_DIM = 256
FUSION_HIDDEN_DIM = 256
