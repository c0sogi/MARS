import os
import torch

# -----------------------------------------------------------------------------
# General Configuration
# -----------------------------------------------------------------------------
SEED = 42
NUM_FOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Using 4 workers is generally safe given 12 vCPUs
NUM_WORKERS = 4

# Debugging flags to control dataset size for rapid iteration if needed
DEBUG = False
DEBUG_SAMPLES = 100

# -----------------------------------------------------------------------------
# Data Paths
# -----------------------------------------------------------------------------
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for this idea
WORKING_DIR = "./working/idea_63"
SUBMISSION_DIR = "./submission"

# Raw JSON files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata CSVs
TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
VAL_META = os.path.join(METADATA_DIR, "val.csv")
TEST_META = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
CACHE_DIR = WORKING_DIR
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 75
LEARNING_RATE = 1e-3  # Constant learning rate as per strategy
PATIENCE = 12  # Early stopping patience
WEIGHT_DECAY = 1e-4  # L2 Regularization
DROPOUT_RATE = 0.5  # Dropout applied after activation

# -----------------------------------------------------------------------------
# Model & Data Specifics
# -----------------------------------------------------------------------------
IMAGE_SIZE = 75
IN_CHANNELS = 3  # HH, HV, and (HH+HV)/2
NUM_CLASSES = 1  # Binary classification (Iceberg vs Ship)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def setup_directories():
    """
    Creates necessary directories for working, caching, checkpoints, and submission.
    """
    dirs_to_create = [WORKING_DIR, CACHE_DIR, CHECKPOINT_DIR, SUBMISSION_DIR]

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    print(f"Configuration loaded. Working directory: {WORKING_DIR}")
