import os
import torch

# =============================================================================
# DIRECTORIES AND PATHS
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_11"
SUBMISSION_DIR = "./submission"

# Raw Data Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Files (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA STATISTICS (GLOBAL MIN/MAX)
# =============================================================================
# Derived from Data Analysis. Used for Global Min-Max Normalization.
BAND_1_MIN = -45.5944
BAND_1_MAX = 32.1806
BAND_2_MIN = -45.6555
BAND_2_MAX = 17.8628

# =============================================================================
# DATA PROCESSING & AUGMENTATION
# =============================================================================
ORIGINAL_IMG_SIZE = 75
IMG_SIZE = 224  # Upsampled size for ResNet-18
IN_CHANNELS = 3  # Band 1 (Norm), Band 2 (Norm), Average (Norm)

# Augmentation Settings
ROTATION_LIMIT = 20  # Degrees (+/-) for continuous rotation

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
BACKBONE_NAME = "resnet18"
PRETRAINED = True
NUM_CLASSES = 1
DROPOUT_RATE = 0.5  # High dropout for regularization in Minimalist Head

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
BATCH_SIZE = 32  # Strict batch size to maintain gradient step volume
LEARNING_RATE = 1e-3  # Initial LR for AdamW
WEIGHT_DECAY = 0.01  # Weight decay for AdamW
NUM_FOLDS = 5  # For Phase 1 Calibration
RANDOM_SEED = 42

# Phase 1: Calibration
CALIBRATION_EPOCHS = 30  # Max epochs to search for optimal stopping
PATIENCE = 5  # For scheduler/early stopping analysis

# Phase 2: Production
NUM_ENSEMBLE_MODELS = 5  # Number of full-fit models to train

# =============================================================================
# HARDWARE & RUNTIME
# =============================================================================
NUM_WORKERS = 4  # Safe number for 12 vCPUs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
