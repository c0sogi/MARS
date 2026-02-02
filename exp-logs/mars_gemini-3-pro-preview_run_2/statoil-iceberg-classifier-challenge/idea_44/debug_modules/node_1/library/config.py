import os
import torch

# ==========================================
# DIRECTORY AND FILE PATHS
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Artifact directory for idea_44 as specified in requirements
WORK_DIR = "./working/idea_44"
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Files
TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
VAL_META = os.path.join(METADATA_DIR, "val.csv")
TEST_META = os.path.join(METADATA_DIR, "test.csv")

# Output Files
CACHE_PATH = os.path.join(WORK_DIR, "processed_data.npz")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# GLOBAL HYPERPARAMETERS
# ==========================================
# Reproducibility
SEED = 42

# Data Dimensions
# 3 Channels: Band 1, Band 2, Average(Band 1, Band 2)
INPUT_SHAPE = (75, 75, 3)

# Training Configuration
NUM_FOLDS = 5
BATCH_SIZE = 32
LEARNING_RATE = 2e-4  # Low learning rate for stability
NUM_EPOCHS = 50  # Maximum epochs, controlled by early stopping
PATIENCE = 10  # Early stopping patience
NUM_WORKERS = 2  # Data loader workers

# Model Architecture Hyperparameters
DROPOUT_RATE = 0.5
VISUAL_FILTERS = 128  # Sustained width for visual backbone
ANCHOR_HIDDEN_UNITS = 16  # Hidden units for Input Anchor Branch
PATH_A_UNITS = 64  # Units for Spatial Context Path
NUM_CLASSES = 1  # Binary classification

# Compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# DEBUGGING & DEVELOPMENT
# ==========================================
# Set to an integer (e.g., 100) to limit dataset size for quick debugging.
# Set to None to use the full dataset.
DEBUG_SAMPLE_SIZE = None
