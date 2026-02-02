import os

# ==========================================
# Directories and Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_12"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission Directory and Path
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Data Configuration
# ==========================================
IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 1
NUM_WORKERS = 4  # Number of data loading workers

# ==========================================
# Model Architecture Configuration
# ==========================================
# Narrow SE-ResNet Backbone
# Channel widths for the three main stages
MODEL_CHANNELS = [16, 32, 64]

# Learnable Pooling (GeM)
# Initial value for the power parameter p
GEM_P_INIT = 3.0

# ==========================================
# Training Hyperparameters
# ==========================================
SEEDS = [0, 1, 2, 3, 4]  # 5 seeds for Homogeneous Ensemble
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Scheduler settings (Cosine Annealing)
T_MAX = EPOCHS
ETA_MIN = 1e-6

# ==========================================
# Debugging / Development
# ==========================================
# Set to True to run on a small subset of data for pipeline verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 100
