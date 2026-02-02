import os
import torch

# ==========================================
# Hyperparameters & Model Configuration
# ==========================================

# Reproducibility
SEED = 42

# Input Dimensions
# Native resolution for EfficientNet-B0 to maximize transfer learning fidelity
IMG_SIZE = 224

# Data Sampling
# Total slices to sample uniformly from the volume (10%-90% depth)
# These are then split into Even (16) and Odd (16) streams.
NUM_SLICES_PER_MODALITY = 32

# Model Architecture
# Channels per stream: 16 slices * 4 modalities = 64 channels
IN_CHANS = 64
BACKBONE = "efficientnet_b0"

# Training
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
N_EPOCHS = 10  # Default number of epochs
DROP_PATH_RATE = 0.2

# Compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2  # Adjust based on vCPU availability (12 vCPUs available)

# ==========================================
# Paths & Directories
# ==========================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"

# Specific Experiment Cache Directory
# Used for storing processed numpy arrays to avoid re-processing
CACHE_DIR = os.path.join(WORKING_DIR, "idea_37")
os.makedirs(CACHE_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.parquet")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.parquet")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.parquet")

# Model & Submission Paths
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Debugging / Development
# ==========================================
# Set to True to run on a small subset of data for quick pipeline verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 16
