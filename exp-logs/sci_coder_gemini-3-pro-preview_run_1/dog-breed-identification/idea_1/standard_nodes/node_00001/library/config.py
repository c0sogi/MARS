import os
import torch

# ==========================================
# Directory and File Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"

# Create working and submission directories if they don't exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Model Artifacts
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "resnet18_best.pth")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
IMAGE_SIZE = 224
NUM_CLASSES = 120
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# Debugging / Development
# Set DEBUG to True to train on a small subset of data for quick verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 100

# ==========================================
# Model Hyperparameters
# ==========================================
MODEL_NAME = "resnet18"
PRETRAINED = True

# ==========================================
# Training Configuration
# ==========================================
SEED = 42
BATCH_SIZE = 64

# Two-phase training strategy
# Phase 1: Train only the head
EPOCHS_HEAD = 5
LEARNING_RATE_HEAD = 1e-3

# Phase 2: Fine-tune the backbone (layer 4 + head)
EPOCHS_FINETUNE = 10
LEARNING_RATE_FINETUNE = 1e-4

# Early Stopping
PATIENCE = 3

# ==========================================
# Compute Configuration
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
