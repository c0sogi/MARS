import os
import torch

# ==========================================
# Path Configuration
# ==========================================
# Root directory for input data (Read-Only)
INPUT_DIR = "./input"

# Directory for pre-computed metadata (Read-Only)
METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Root directory for working outputs (Write-Enabled)
# Using idea_29 as specified in the prompt logic
WORKING_DIR = "./working/idea_29"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure working directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Processing Configuration
# ==========================================
# Cutoff radius for neighbor search in Angstroms
# Prioritizing a clean Radius Graph to preserve local atomic density
CUTOFF_RADIUS = 5.0

# Number of Gaussian Radial Basis Function bins for edge expansion
NUM_RBF = 60

# Maximum number of neighbors to consider per node (for efficiency)
# Though we use a radius graph, setting a high max neighbors prevents memory OOM on dense structures
MAX_NEIGHBORS = 50

# Random seed for reproducibility
SEED = 42

# ==========================================
# Model Architecture Configuration
# ==========================================
# Dimension of node embeddings and hidden layers
HIDDEN_DIM = 128

# Number of interaction blocks in the backbone
NUM_LAYERS = 4

# Dropout rate applied within interaction blocks and prediction heads
DROPOUT = 0.1

# ==========================================
# Training Configuration
# ==========================================
# Batch size for training and evaluation
# Selected to introduce beneficial gradient noise
BATCH_SIZE = 48

# Initial learning rate for the optimizer
LEARNING_RATE = 1e-3

# Weight decay for regularization (AdamW)
WEIGHT_DECAY = 1e-4

# Maximum number of training epochs
MAX_EPOCHS = 150

# Patience for learning rate scheduler (ReduceLROnPlateau)
SCHEDULER_PATIENCE = 10

# Factor by which to reduce learning rate
SCHEDULER_FACTOR = 0.5

# Patience for early stopping
EARLY_STOPPING_PATIENCE = 25

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Number of workers for data loading
NUM_WORKERS = 4
