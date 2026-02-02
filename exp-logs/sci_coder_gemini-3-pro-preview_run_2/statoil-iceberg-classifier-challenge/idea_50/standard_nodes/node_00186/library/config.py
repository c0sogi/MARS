import os
import torch

# ==========================================
# PATH CONFIGURATION
# ==========================================
INPUT_DIR = "./input"
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Working directory for the specific idea implementation
WORK_DIR = "./working/idea_50"
CACHE_DIR = os.path.join(WORK_DIR, "cache")
MODEL_DIR = os.path.join(WORK_DIR, "models")
SUBMISSION_DIR = "./submission"
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# DATA & MODEL HYPERPARAMETERS
# ==========================================
SEED = 42
IMAGE_SIZE = 75
IN_CHANNELS = 3  # Band 1, Band 2, Mean
NUM_CLASSES = 1  # Binary classification

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
NUM_FOLDS = 5
BATCH_SIZE = 32
LEARNING_RATE = 2e-4  # "Low and Slow" strategy
MAX_EPOCHS = 100
PATIENCE = 15  # Early stopping patience
NUM_WORKERS = 4  # Number of dataloader workers

# ==========================================
# COMPUTE CONFIGURATION
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
