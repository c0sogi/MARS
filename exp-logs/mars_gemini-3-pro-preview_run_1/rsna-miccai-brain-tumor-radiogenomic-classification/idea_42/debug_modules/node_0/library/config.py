import os
import torch

# ==========================================
# Global Constants & Paths
# ==========================================
SEED = 42

# Directory Paths
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Working Directory for Caching (Idea 42)
# Stores processed numpy arrays/parquets to avoid re-computing ROI/CoM
WORKING_DIR = "./working/idea_42"
os.makedirs(WORKING_DIR, exist_ok=True)

# Output Paths
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Hyperparameters
# ==========================================
IMG_SIZE = 224
NUM_CHANNELS = 3  # FLAIR, T1wCE, T2w (Early Fusion)
NUM_FOLDS = 5

# The specific modalities used in the 3-channel stack
# Note: T1w is excluded based on the strategy description
SELECTED_MODALITIES = ["FLAIR", "T1wCE", "T2w"]

# ==========================================
# Model Hyperparameters
# ==========================================
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
NUM_CLASSES = 1  # Binary Classification
DROPOUT_RATE = 0.3

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive regularization
NUM_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 4

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Caching Filenames
# ==========================================
# These files will be saved in WORKING_DIR
CACHE_TRAIN_IMAGES = "train_images_roi.npy"
CACHE_TRAIN_LABELS = "train_labels.npy"
CACHE_VAL_IMAGES = "val_images_roi.npy"
CACHE_VAL_LABELS = "val_labels.npy"
CACHE_TEST_IMAGES = "test_images_roi.npy"
CACHE_TEST_IDS = "test_ids.npy"
