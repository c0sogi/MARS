import os
import torch

# =============================================================================
# DIRECTORY AND FILE PATHS
# =============================================================================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, "input")
METADATA_DIR = os.path.join(BASE_DIR, "metadata")
WORKING_DIR = os.path.join(BASE_DIR, "working", "idea_22")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data Paths
SPECTROGRAM_DIR = os.path.join(INPUT_DIR, "supplemental_data", "spectrograms")
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache Paths for deterministic loading
CACHE_DIR = WORKING_DIR  # Cache directly in the working directory
TRAIN_CACHE_PATH = os.path.join(CACHE_DIR, "train_cache.parquet")
VAL_CACHE_PATH = os.path.join(CACHE_DIR, "val_cache.parquet")
TEST_CACHE_PATH = os.path.join(CACHE_DIR, "test_cache.parquet")

# Checkpoint Paths
TEACHER_CHECKPOINT_TEMPLATE = os.path.join(WORKING_DIR, "teacher_fold_{}.pth")
STUDENT_CHECKPOINT_PATH = os.path.join(WORKING_DIR, "student_swa.pth")
PSEUDO_LABELS_PATH = os.path.join(WORKING_DIR, "pseudo_labels.parquet")

# =============================================================================
# DATA HYPERPARAMETERS
# =============================================================================
NUM_CLASSES = 19
IN_CHANNELS = 3  # Channel replication for ResNet

# Image Dimensions
IMG_HEIGHT = 256
IMG_WIDTH_TEST = 640  # Center of the high-fidelity band for standard inference

# Dynamic Temporal Jittering (Training)
# Widths will be sampled from this range during training
JITTER_RANGE = (576, 704)

# Multi-Scale TTA (Inference/Pseudo-labeling)
# Fixed widths for test-time augmentation
TTA_WIDTHS = [608, 640, 672]

# Normalization (ImageNet Statistics)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Augmentation
MIXUP_ALPHA = 0.2

# Stochastic Weight Averaging (SWA)
SWA_START_EPOCH = 38
SWA_LR = 1e-4  # Standard practice to lower LR for SWA, though not explicitly fixed in prompt text

# Debugging / Development
# Set to None to use full dataset, or an integer (e.g., 50) to limit dataset size
DEBUG_MAX_SAMPLES = None

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
