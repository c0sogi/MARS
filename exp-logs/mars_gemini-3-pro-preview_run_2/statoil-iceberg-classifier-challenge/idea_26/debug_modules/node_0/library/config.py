import os
import torch

# ==========================================
# DIRECTORY & FILE PATHS
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_26"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Raw Data Files
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata Files (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Files
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# ==========================================
# DATA CONFIGURATION
# ==========================================
IMG_HEIGHT = 75
IMG_WIDTH = 75
IN_CHANNELS = 3  # Band 1, Band 2, Mean(B1, B2)

# Augmentation Settings
# Strategy: Rotation (0, 90, 180, 270) + Horizontal Flip. No Vertical Flip.
AUGMENT_ROTATION = True
AUGMENT_HFLIP = True
AUGMENT_VFLIP = False
AUGMENT_MIXUP = False

# Debugging / Development
# Set to an integer (e.g., 100) to limit dataset size for fast debugging. Set to None for full run.
DEBUG_DATA_SIZE = None

# ==========================================
# MODEL CONFIGURATION
# ==========================================
MODEL_NAME = "IDSW_Net"
BASE_FILTERS = 128
DROPOUT_RATE = 0.5  # High dropout to regularize wide backbone
USE_INC_ANGLE = True

# ==========================================
# TRAINING CONFIGURATION
# ==========================================
RANDOM_SEED = 42
N_FOLDS = 5

# Optimization
BATCH_SIZE = 32
NUM_EPOCHS = 60  # High enough to allow "Low and Slow" convergence
LEARNING_RATE = 2e-4  # Conservative start
WEIGHT_DECAY = 0.0  # Rely on Dropout and Augmentation first

# Scheduler & Early Stopping
PATIENCE = 12  # Patience for Early Stopping
SCHEDULER_PATIENCE = 5  # Patience for ReduceLROnPlateau
SCHEDULER_FACTOR = 0.5  # Decay factor
MIN_LR = 1e-6

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Number of dataloader workers
