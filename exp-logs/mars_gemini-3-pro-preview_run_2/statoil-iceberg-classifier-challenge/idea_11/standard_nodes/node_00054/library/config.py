import os

# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_11"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# 2. DATA PATHS
# ==========================================
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")

TRAIN_META_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_META_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_META_CSV = os.path.join(METADATA_DIR, "test.csv")

SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache path for processed data (numpy format)
CACHE_PATH = os.path.join(WORKING_DIR, "processed_data.npz")

# ==========================================
# 3. HYPERPARAMETERS
# ==========================================
SEED = 42
N_FOLDS = 5
BATCH_SIZE = 32
NUM_EPOCHS = 100  # Upper bound; early stopping is expected to intervene
LEARNING_RATE = 2e-4
PATIENCE = 20  # For EarlyStopping and Scheduler (Cite solution_lesson_node_00023)
MIN_LR = 1e-6  # Minimum learning rate for ReduceLROnPlateau

# ==========================================
# 4. MODEL CONFIGURATION (WEBN)
# ==========================================
# Input image specs
IMG_HEIGHT = 75
IMG_WIDTH = 75
IMG_CHANNELS = 3  # Band 1, Band 2, Avg

# Backbone Architecture: Expansion-Compression
# Stage 1 (Stem): 64 filters
# Stage 2 (Expansion): 128 filters
# Stage 3 (Peak): 256 filters
# Stage 4 (Bottleneck): 48 filters
BACKBONE_FILTERS = [64, 128, 256, 48]

# Regularization
DROPOUT_RATE = 0.2

# ==========================================
# 5. OUTPUT PATHS
# ==========================================
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")


def get_model_path(fold_idx):
    """Returns the path to save/load the model for a specific fold."""
    return os.path.join(WORKING_DIR, f"webn_model_fold_{fold_idx}.pth")
