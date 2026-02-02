import os
import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_13"

# Raw Data Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Files (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Directories
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = os.path.join(".", "submission")

# Ensure output directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Final Submission File
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# DATA STATISTICS & PREPROCESSING
# =============================================================================

# Global Statistics for Min-Max Normalization (Derived from Data Analysis)
# Band 1 (HH)
BAND1_MIN = -45.5944
BAND1_MAX = 32.1806

# Band 2 (HV)
BAND2_MIN = -45.6555
BAND2_MAX = 17.8628

# Image Transformations
ORIGINAL_SIZE = 75
IMAGE_SIZE = 224  # Upsampling target size
INTERPOLATION = "bicubic"
ROTATION_RANGE = 20  # Continuous rotation +/- degrees
DO_HORIZONTAL_FLIP = True
DO_VERTICAL_FLIP = True

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

MODEL_NAME = "resnet18"
USE_PRETRAINED = True
NUM_CLASSES = 1
DROPOUT_RATE = 0.5
IN_CHANNELS = 3  # Band 1, Band 2, Composite (Mean)
FEAT_DIM = 512  # ResNet18 Global Average Pooling dimension

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

# General
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
NUM_FOLDS = 5

# Optimization (AdamW)
BATCH_SIZE = 32
LEARNING_RATE = 2e-4  # Cite solution_lesson_node_00009
WEIGHT_DECAY = 0.01

# Phase 1: Calibration (Finding Convergence Epoch)
MAX_EPOCHS_PHASE_1 = 75
PATIENCE = 10  # Early stopping patience

# Phase 2: Full-Fit SWA (Stochastic Weight Averaging)
# The model trains for E_conv epochs, then enters SWA phase
SWA_EPOCHS = 12  # Number of epochs to perform averaging
SWA_LR = 1e-4  # Constant learning rate for SWA phase
SWA_START_FACTOR = (
    1.0  # Factor of E_conv to start SWA (usually 1.0 = immediately after)
)
