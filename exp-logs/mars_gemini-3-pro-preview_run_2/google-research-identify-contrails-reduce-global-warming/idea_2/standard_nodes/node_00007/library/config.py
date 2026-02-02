import os
import torch

# ==========================================
# Directory and File Paths
# ==========================================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
WORKING_DIR = os.path.join(BASE_DIR, "working")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VALIDATION_METADATA_PATH = os.path.join(METADATA_DIR, "validation_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cache Directory for Idea 2 (Symmetric Temporal-Difference)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_2")
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# Model Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 12
IMG_SIZE = 256

# Input Specifications for U-Net
# 3 channels (Current) + 3 channels (Past Diff)
N_CHANNELS = 6
N_CLASSES = 1  # Binary segmentation

# Encoder Backbone
ENCODER_NAME = "resnet18"
ENCODER_WEIGHTS = "imagenet"

# ==========================================
# Data Processing Constants
# ==========================================
# Ash Composite Construction
# Red: Band 15 - Band 14
# Green: Band 14 - Band 11
# Blue: Band 14
ASH_BAND_IDS = [11, 14, 15]

# Temporal Indices (0-indexed)
# Sequence length is usually 8 (t=0 to t=7). Labeled frame is at index 4.
TIME_CURRENT = 4
TIME_PREV = 3
TIME_NEXT = 5

# Normalization Constants (Approximate for brightness temperature differences)
# These can be tuned based on EDA, but standard ranges for Ash composite are often used.
# Storing as (min, max) for min-max scaling or similar.
# Specific calibration might be handled in the dataset class.

# ==========================================
# Compute Resources
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Adjustable based on vCPU count (12 available)

# ==========================================
# Debugging / Development
# ==========================================
# Set to True to run on a small subset of data for quick pipeline testing
DEBUG = False
DEBUG_SAMPLE_SIZE = 500
