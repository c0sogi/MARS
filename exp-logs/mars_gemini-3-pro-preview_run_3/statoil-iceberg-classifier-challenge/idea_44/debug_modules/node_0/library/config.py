import os

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# Directory Paths
# ==========================================
# Input data (Read-Only)
INPUT_DIR = "./input"
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata (Pre-generated)
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working Directory for Cache and Intermediate Files
# Using specific subdirectory for this idea
WORK_DIR = "./working/idea_44"
CACHE_DIR = WORK_DIR  # Where .npy files will be stored

# Checkpoints
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

# Submission Output
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure writable directories exist
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Data Configuration
# ==========================================
IMAGE_SIZE = 75
NUM_BANDS_RAW = 2  # HH, HV
NUM_INPUT_CHANNELS = 3  # HH, HV, Average(HH, HV)

# ==========================================
# Model Hyperparameters
# ==========================================
# Architecture specific
LEAKY_RELU_SLOPE = 0.1
DROPOUT_RATE = 0.5

# ==========================================
# Training Hyperparameters
# ==========================================
NUM_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 75
LEARNING_RATE = 1e-3  # Constant learning rate
WEIGHT_DECAY = 1e-4  # L2 Regularization
PATIENCE = 12  # Early stopping patience
