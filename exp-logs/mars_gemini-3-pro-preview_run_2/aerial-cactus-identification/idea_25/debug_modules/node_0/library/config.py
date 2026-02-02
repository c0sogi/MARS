import os

# ==========================================
#              Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_25"
SUBMISSION_DIR = "./submission"

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Output Paths
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
#              Data Configuration
# ==========================================
IMAGE_SIZE = 32
NUM_CLASSES = 1  # Binary classification (Cactus vs No Cactus)
NUM_WORKERS = 4  # Optimized for 12 vCPUs

# ==========================================
#           Model Architecture
# ==========================================
# "Super-Wide" Channel Configuration as per strategy
MODEL_CHANNELS = [64, 128, 256]

# Grouped Convolution Cardinality
CARDINALITY = 32

# ==========================================
#           Training Configuration
# ==========================================
# Homogeneous Seed Averaging
SEEDS = [0, 1, 2, 3, 4]

EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Early Stopping
PATIENCE = 5

# ==========================================
#           Inference Configuration
# ==========================================
# Test Time Augmentation (Horizontal and Vertical Flips)
TTA_ENABLED = True

# Debug Mode (Set to True to run on a small subset for testing pipeline)
DEBUG = False
