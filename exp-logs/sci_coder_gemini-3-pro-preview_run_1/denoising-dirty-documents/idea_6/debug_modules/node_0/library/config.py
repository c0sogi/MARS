import os
import torch

# =============================================================================
# GLOBAL SEED
# =============================================================================
SEED = 42

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
# Input directories (Read-Only)
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working directory for caching and model checkpoints
# Specific to the current idea iteration
WORKING_DIR = "./working/idea_6"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# DATASET & DATALOADER PARAMETERS
# =============================================================================
PATCH_SIZE = 160  # Training on 160x160 random crops
BATCH_SIZE = 16  # Small batch size for optimization velocity
NUM_WORKERS = 4  # Number of subprocesses for data loading
PIN_MEMORY = True  # Pin memory for faster host-to-device transfer

# =============================================================================
# MODEL PARAMETERS
# =============================================================================
# Shallow U-Net Configuration
IN_CHANNELS = 1  # Grayscale input
OUT_CHANNELS = 1  # Grayscale output
BASE_FILTERS = 32  # Starting filter count (32 -> 64 -> 128)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
NUM_FOLDS = 5  # 5-Fold Cross-Validation
EPOCHS = 1000  # Full convergence training
LEARNING_RATE = 1e-3  # High initial learning rate
WEIGHT_DECAY = 0.0  # Standard weight decay (if needed)

# Scheduler Parameters
T_MAX = EPOCHS  # Cosine Annealing T_max matches total epochs
ETA_MIN = 0.0  # Minimum learning rate

# =============================================================================
# HARDWARE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# INFERENCE PARAMETERS
# =============================================================================
TTA_VIEWS = 8  # Number of Test-Time Augmentation views (flips/rotations)
