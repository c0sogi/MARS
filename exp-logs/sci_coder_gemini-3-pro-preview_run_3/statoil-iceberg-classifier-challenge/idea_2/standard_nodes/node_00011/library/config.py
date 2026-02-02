import os
import torch

# ==========================================
# General Configuration
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Directory Paths
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_2"
SUBMISSION_DIR = "./submission"

# Ensure working and submission directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# File Paths
# ==========================================
# Raw Data
TRAIN_JSON = os.path.join(INPUT_DIR, "train.json")
TEST_JSON = os.path.join(INPUT_DIR, "test.json")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata
TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
VAL_META = os.path.join(METADATA_DIR, "val.csv")
TEST_META = os.path.join(METADATA_DIR, "test.csv")

# Submission Output
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# Caching Paths (Numpy format)
# ==========================================
# We define specific paths for cached numpy arrays to be used by the data loader
CACHE_X_TRAIN = os.path.join(WORKING_DIR, "X_train.npy")
CACHE_Y_TRAIN = os.path.join(WORKING_DIR, "y_train.npy")
CACHE_ANGLE_TRAIN = os.path.join(WORKING_DIR, "angle_train.npy")

CACHE_X_VAL = os.path.join(WORKING_DIR, "X_val.npy")
CACHE_Y_VAL = os.path.join(WORKING_DIR, "y_val.npy")
CACHE_ANGLE_VAL = os.path.join(WORKING_DIR, "angle_val.npy")

CACHE_X_TEST = os.path.join(WORKING_DIR, "X_test.npy")
CACHE_ANGLE_TEST = os.path.join(WORKING_DIR, "angle_test.npy")
CACHE_ID_TEST = os.path.join(WORKING_DIR, "test_ids.npy")

# ==========================================
# Data Parameters
# ==========================================
ORIGINAL_IMAGE_SIZE = 75
# Upscaling to 224 for VGG16 compatibility
IMAGE_SIZE = 224
# 3 Channels: HH, HV, Average(HH, HV)
INPUT_CHANNELS = 3
NUM_CLASSES = 1

# ==========================================
# Training Hyperparameters
# ==========================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 7
N_FOLDS = 5

# ==========================================
# Model Specifics
# ==========================================
# Feature dimension after VGG16 features (512 channels) + Global Max Pooling
BACKBONE_OUT_DIM = 512
# Dimension after concatenating incidence angle
FUSION_DIM = BACKBONE_OUT_DIM + 1
