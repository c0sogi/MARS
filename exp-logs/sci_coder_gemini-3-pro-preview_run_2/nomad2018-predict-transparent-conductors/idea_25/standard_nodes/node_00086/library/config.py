import os

# ==========================================
# Directory and File Paths
# ==========================================
ROOT_DIR = "."
INPUT_DIR = os.path.join(ROOT_DIR, "input")
METADATA_DIR = os.path.join(ROOT_DIR, "metadata")
WORKING_DIR = os.path.join(ROOT_DIR, "working", "idea_25")
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(ROOT_DIR, "submission")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata file paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# ==========================================
# Data & Graph Construction Parameters
# ==========================================
CUTOFF_RADIUS = 5.0  # Interaction radius in Angstroms
NUM_RBF_BINS = 60  # Number of Gaussian RBF bins for edge expansion
MAX_NEIGHBORS = (
    50  # Maximum number of neighbors per atom (for fixed-size tensor support)
)
RBF_GAMMA = 10.0  # Gamma parameter for RBF kernel

# ==========================================
# Model Architecture Hyperparameters
# ==========================================
HIDDEN_DIM = 128  # Dimension of node embeddings and hidden layers
NUM_BLOCKS = 4  # Number of interaction blocks
DROPOUT_RATE = 0.1  # Dropout probability
ATOM_EMBEDDING_DIM = 128  # Input embedding dimension for atomic numbers
EDGE_EMBEDDING_DIM = 128  # Projected dimension for edge features

# ==========================================
# Training Configuration
# ==========================================
RANDOM_SEED = 42
BATCH_SIZE = 48
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4  # Decoupled weight decay for AdamW
NUM_EPOCHS = 120  # Total number of training epochs
EARLY_STOPPING_PATIENCE = (
    20  # Epochs to wait before stopping if val loss doesn't improve
)

# Scheduler settings (ReduceLROnPlateau)
SCHEDULER_FACTOR = 0.6
SCHEDULER_PATIENCE = 8
SCHEDULER_MIN_LR = 1e-6

# ==========================================
# Target Variables
# ==========================================
TARGET_COLS = ["formation_energy_ev_natom", "bandgap_energy_ev"]

# ==========================================
# Debugging / Development
# ==========================================
# Set to True to use a smaller subset of data for rapid testing
DEBUG_MODE = False
DEBUG_DATA_SIZE = 100  # Number of samples to use in debug mode
