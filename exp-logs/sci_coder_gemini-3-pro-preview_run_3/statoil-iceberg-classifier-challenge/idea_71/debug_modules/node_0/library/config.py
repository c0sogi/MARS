import os

# =============================================================================
# Global Random Seed
# =============================================================================
SEED = 42

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 75
PATIENCE = 12
NUM_FOLDS = 5

# =============================================================================
# Directory Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_71"

# Derived Directories for Artifacts
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# =============================================================================
# File Paths
# =============================================================================
# Raw Data
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata
TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
VAL_META = os.path.join(METADATA_DIR, "val.csv")
TEST_META = os.path.join(METADATA_DIR, "test.csv")

# =============================================================================
# Directory Initialization
# =============================================================================
# Ensure all necessary working directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
