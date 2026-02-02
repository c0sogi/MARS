import os
import torch

# ==========================================
# GLOBAL SETTINGS
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# ==========================================
# DIRECTORY SETUP
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Specific working directory for this experiment idea
WORKING_DIR = "./working/idea_51"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# FILE PATHS
# ==========================================
# Raw JSON inputs
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Generated Metadata CSVs
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Artifacts
PROCESSED_DATA_CACHE = os.path.join(WORKING_DIR, "processed_data.npz")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
CHECKPOINT_DIR = WORKING_DIR

# ==========================================
# DATA PARAMETERS
# ==========================================
IMAGE_SIZE = 75
NUM_CHANNELS = 3  # Band 1, Band 2, Average(B1, B2)
# Set to an integer (e.g., 100) to limit dataset size for debugging, or None for full run
DEBUG_MAX_SAMPLES = None

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
# Wide-Body Backbone: Sustained width strategy
NUM_FILTERS = 128
# Regularization
DROPOUT_RATE = 0.5

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
NUM_FOLDS = 5
BATCH_SIZE = 64
# "Low and Slow" optimization strategy
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
# Early Stopping
PATIENCE = 15
# Learning Rate Scheduler (ReduceLROnPlateau)
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-6
