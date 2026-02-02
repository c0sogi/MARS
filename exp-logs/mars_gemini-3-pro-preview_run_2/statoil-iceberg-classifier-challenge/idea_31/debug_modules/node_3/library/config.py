import os
import torch

# ==========================================
# 1. PATHS & DIRECTORIES
# ==========================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_31"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Files (JSON)
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Files (CSV)
TRAIN_META_FILE = os.path.join(METADATA_DIR, "train.csv")
VAL_META_FILE = os.path.join(METADATA_DIR, "val.csv")
TEST_META_FILE = os.path.join(METADATA_DIR, "test.csv")

# Processed Data Cache
# Used to store the numpy/parquet cache of the processed images
PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

# Model Artifacts
# Directory to save model checkpoints during training
MODEL_CHECKPOINT_DIR = WORKING_DIR
# Final submission file path
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# 2. HYPERPARAMETERS
# ==========================================
SEED = 42
NUM_FOLDS = 5
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
PATIENCE = 15  # Early stopping patience

# ==========================================
# 3. DATA SPECIFICATIONS
# ==========================================
IMAGE_SIZE = 75
NUM_CHANNELS = 3  # Constructed from: Band 1, Band 2, Mean(B1, B2)
NUM_CLASSES = 1  # Binary classification (0=Ship, 1=Iceberg)

# ==========================================
# 4. MODEL ARCHITECTURE (RDP-WBN)
# ==========================================
# Wide-Body Backbone settings
BACKBONE_FILTERS = 128
# Dual-Path Readout settings
READOUT_PATH_A_FILTERS = 48  # Spatial Context path
# Regularization
DROPOUT_RATE = 0.5

# ==========================================
# 5. COMPUTE CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 12 vCPUs available, using a safe number for workers
NUM_WORKERS = 4
