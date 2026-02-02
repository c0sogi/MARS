import os
import torch

# ====================================================
# General Settings
# ====================================================
SEED = 42
DEBUG = False  # Set to True to run with a small subset of data for debugging
N_DEBUG_SAMPLES = 50  # Number of samples to use when DEBUG is True
EXPERIMENT_NAME = "idea_12"

# ====================================================
# Directory Paths
# ====================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = os.path.join("./working", EXPERIMENT_NAME)
SUBMISSION_DIR = "./submission"

# Input Data Files
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Directories (Created automatically)
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure necessary directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ====================================================
# Data Hyperparameters
# ====================================================
IMG_SIZE = 260
NUM_SLICES = 3  # Apical, Middle, Basal
IN_CHANNELS = 3  # 3 slices stacked as channels

# Target Normalization (Calculated from Training EDA)
# Used for Z-score standardization of the target FVC
TARGET_MEAN = 2654.6528
TARGET_STD = 801.7017

# Feature Normalization
TIME_SCALE = 0.01  # Scaling factor for relative time (Weeks)

# Metric Constants
MIN_UNCERTAINTY = 70  # sigma_clipped lower bound
MAX_ERROR = 1000  # delta upper bound

# ====================================================
# Model Hyperparameters
# ====================================================
BACKBONE_NAME = "tf_efficientnet_b2"
FEATURE_DIM = 128
DROP_RATE = 0.0  # Explicitly 0.0 as per RSTC-Net design
DO_BATCHNORM = False  # Explicitly False for the head as per RSTC-Net design

# ====================================================
# Training Hyperparameters
# ====================================================
EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 4

# Optimizer Settings
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-2

# Scheduler Settings (Cosine Annealing)
T_MAX = EPOCHS
ETA_MIN = 1e-6

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
