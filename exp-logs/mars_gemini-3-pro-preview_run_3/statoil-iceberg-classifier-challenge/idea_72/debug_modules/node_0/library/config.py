import os

# ==========================================
# Directory Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific directory for this experiment idea (Idea 72)
# This is used for caching processed data and saving model checkpoints
IDEA_ID = "idea_72"
IDEA_DIR = os.path.join(WORKING_DIR, IDEA_ID)
CACHE_DIR = IDEA_DIR  # Cache files (npy) go here
CHECKPOINT_DIR = os.path.join(IDEA_DIR, "checkpoints")

# Ensure working directories exist
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# File Paths
# ==========================================
# Raw Data
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata (Generated previously)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# ==========================================
# Data Parameters
# ==========================================
IMAGE_HEIGHT = 75
IMAGE_WIDTH = 75
NUM_BANDS_RAW = 2  # HH, HV
NUM_CHANNELS_MODEL = 3  # HH, HV, Average (Synthetic)

# ==========================================
# Training Hyperparameters
# ==========================================
SEED = 42
NUM_FOLDS = 5
BATCH_SIZE = 32
EPOCHS = 75
LEARNING_RATE = 1e-3  # Constant LR as per strategy
PATIENCE = 12  # Early stopping patience
WEIGHT_DECAY = 1e-2  # L2 Regularization for AdamW (standard default)
NUM_WORKERS = 4  # For DataLoader

# ==========================================
# Model Architecture Parameters
# ==========================================
# Specific to Tri-Statistic Isomorphic CNN
DROPOUT_RATE = 0.5
ATTENTION_REDUCTION_RATIO = 16
PROJECTION_DIM = 42  # Project 128 channels -> 42 before tri-stat pooling
HIDDEN_DIM = 256  # Classification head hidden size
