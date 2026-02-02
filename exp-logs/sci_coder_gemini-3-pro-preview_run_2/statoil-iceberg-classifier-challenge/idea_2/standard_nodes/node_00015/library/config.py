import os

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Specific file paths
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache paths for processed data
PROCESSED_DATA_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

# Model checkpoint path
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "sa_hcn_model.pth")

# Submission output path
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# 2. DATA & MODEL HYPERPARAMETERS
# ==========================================
# Image dimensions
IMAGE_SIZE = 75
CHANNELS = 3  # Band 1, Band 2, Average(Band 1, Band 2)

# Normalization parameters (approximate min/max from EDA for scaling to [0, 1])
# Band 1 range: ~ -45 to +32
# Band 2 range: ~ -45 to +17
# We use a slightly wider range to be safe and consistent
MIN_DB = -50.0
MAX_DB = 40.0

# ==========================================
# 3. TRAINING HYPERPARAMETERS
# ==========================================
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
NUM_EPOCHS = 100
PATIENCE = 20  # For early stopping

# Dropout rate for the dense layers to control overfitting
# Reduced to 0.2 as stronger augmentation (rotation) provides regularization
DROPOUT_RATE = 0.2

# ==========================================
# 4. DEBUGGING
# ==========================================
# Set to True to run on a small subset of data for quick pipeline testing
DEBUG = False
DEBUG_SIZE = 100
