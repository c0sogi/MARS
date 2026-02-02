import os
import torch

# ==========================================
# Path Configuration
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_30"

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")
SUBMISSION_PATH = "./submission/submission.csv"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Cache Files
CACHE_TRAIN_IMAGES = os.path.join(WORKING_DIR, "train_images.npy")
CACHE_TRAIN_LABELS = os.path.join(WORKING_DIR, "train_labels.npy")
CACHE_VAL_IMAGES = os.path.join(WORKING_DIR, "val_images.npy")
CACHE_VAL_LABELS = os.path.join(WORKING_DIR, "val_labels.npy")
CACHE_TEST_IMAGES = os.path.join(WORKING_DIR, "test_images.npy")
CACHE_TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

# ==========================================
# Data Configuration
# ==========================================
IMG_SIZE = 224
NUM_CHANNELS = 9  # 3 modalities * 3 depths

# Modalities to use (Order matters for channel stacking)
# We use FLAIR, T1wCE, T2w as per the SIRV strategy
MODALITIES = ["FLAIR", "T1wCE", "T2w"]

# Relative Anatomical Depths for sampling
# 0.4 = 40% depth, 0.5 = Center, 0.6 = 60% depth
RELATIVE_DEPTHS = [0.4, 0.5, 0.6]

# ==========================================
# Model Configuration
# ==========================================
BACKBONE = "efficientnet_b0"
PRETRAINED = True
NUM_CLASSES = 1

# Regularization
DROPOUT_RATE = 0.3  # Classifier dropout
DEPTH_DROPOUT_PROB = 0.2  # Input-level structured dropout

# Gaussian Weight Inflation Initialization Factors
# Center slices (channels 3-5) get 50% energy
# Peripheral slices (channels 0-2, 6-8) get 25% energy
WEIGHT_INIT_CENTER = 0.5
WEIGHT_INIT_PERIPHERAL = 0.25

# ==========================================
# Training Configuration
# ==========================================
SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
NUM_FOLDS = 5

# Optimizer Settings
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Aggressive weight decay

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4

# ==========================================
# Augmentation Configuration
# ==========================================
# Note: Translation and Scaling are strictly prohibited to preserve
# the spatial priors established by centroid alignment.
AUG_ROTATION_LIMIT = 15
AUG_ELASTIC_ALPHA = 1.0
AUG_ELASTIC_SIGMA = 50.0
AUG_ELASTIC_ALPHA_AFFINE = 50.0
AUG_GRID_DISTORT_NUM_STEPS = 5
AUG_GRID_DISTORT_DISTORT_LIMIT = 0.3
