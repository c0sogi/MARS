import os

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# Base Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
# Using idea_13 as the working directory for this iteration
WORKING_DIR = "./working/idea_13"
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Raw Data Paths
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

# Metadata Paths (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

# ==========================================
# 2. DATA CONFIGURATION
# ==========================================
IMAGE_SIZE = 75
NUM_CHANNELS = 3  # Band 1 (HH), Band 2 (HV), Average
NUM_CLASSES = 1  # Binary classification (0: Ship, 1: Iceberg)

# Augmentation Settings
# Rotations restricted to 90-degree increments to preserve grid fidelity
ROTATION_ANGLES = [0, 90, 180, 270]
USE_HORIZONTAL_FLIP = True
USE_VERTICAL_FLIP = False  # Excluded to avoid redundancy

# ==========================================
# 3. MODEL CONFIGURATION
# ==========================================
MODEL_NAME = "WBPA_Net"  # Wide-Body Projected Attention Network

# Architecture Hyperparameters
INITIAL_FILTERS = 64
DEEP_FILTERS = 128  # Sustained width for deeper layers
PROJECTION_DIM = 64  # 1x1 projection bottleneck dimension
DROPOUT_RATE = 0.2  # Moderate dropout

# ==========================================
# 4. TRAINING CONFIGURATION
# ==========================================
SEED = 42
NUM_FOLDS = 5  # Stratified 5-Fold Cross-Validation
BATCH_SIZE = 32
NUM_EPOCHS = 50  # Upper bound, controlled by Early Stopping
LEARNING_RATE = 2e-4  # "Low and Slow" initialization
PATIENCE = 10  # Early Stopping patience
NUM_WORKERS = 4  # Data loader workers (12 vCPUs available)
