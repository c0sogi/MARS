import os
import torch

# ==========================================
# Global Configuration for Steganography Detection
# ==========================================

# --- Reproducibility ---
SEED = 42

# --- File Paths ---
INPUT_ROOT = "./input"
METADATA_DIR = "./metadata"

# Metadata CSVs
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Image Directories
# Note: Metadata 'image_path' columns are relative to INPUT_ROOT
# e.g., "Cover/00001.jpg" or "Test/0001.jpg"
IMG_ROOT = INPUT_ROOT

# Output Directories
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_1")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# --- Data Parameters ---
IMAGE_SIZE = 512
IN_CHANNELS = 1  # Model uses only the Luminance (Y) channel
NUM_CLASSES = 1  # Binary classification (Cover vs. Stego)
USE_CACHE = True  # Flag to enable/disable data caching (e.g., for processed arrays)

# --- Training Hyperparameters ---
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 3

# Debugging / Development
DEBUG = False  # Set to True to run on a small subset
DEBUG_SAMPLE_SIZE = 2000

# --- Evaluation / Metric Parameters ---
# Weighted AUC parameters as defined in the task
TPR_THRESHOLDS = [0.0, 0.4, 1.0]
AUC_WEIGHTS = [2, 1]

# Test Time Augmentation
TTA_VIEWS = 4  # Original + 3 rotations (90, 180, 270)

# --- System Settings ---
NUM_WORKERS = 8  # Optimized for 12 vCPUs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_directories():
    """
    Ensures that necessary working and submission directories exist.
    """
    os.makedirs(IDEA_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)


# Initialize directories upon module import
setup_directories()
