import os
import torch

# ==========================================
# Directory and File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Ensure necessary writable directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Input Files
MEGADETECTOR_PATH = os.path.join(INPUT_DIR, "iwildcam2020_megadetector_results.json")
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Files
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
# The sample submission indicates Category IDs range from 0 to 675.
# We set NUM_CLASSES to 676 to handle direct indexing (0-675).
NUM_CLASSES = 676
IMAGE_SIZE = 224
NUM_WORKERS = 8  # Optimized for 12 vCPUs

# ==========================================
# Model Configuration
# ==========================================
MODEL_NAME = "resnet18"
PRETRAINED = True

# ==========================================
# Training Configuration
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 15
SEED = 42
PATIENCE = 4  # Early stopping patience

# ==========================================
# Hardware Configuration
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
