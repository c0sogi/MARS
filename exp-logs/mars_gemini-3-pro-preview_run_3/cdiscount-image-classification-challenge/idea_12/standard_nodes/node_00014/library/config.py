import os
import torch

# ==========================================
# SYSTEM & HARDWARE
# ==========================================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 10  # Utilizing available vCPUs (12 total, leaving 2 overhead)

# ==========================================
# FILE PATHS
# ==========================================
# Input Directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Raw Data
TRAIN_BSON = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON = os.path.join(INPUT_DIR, "test.bson")
CATEGORY_NAMES = os.path.join(INPUT_DIR, "category_names.csv")
SAMPLE_SUBMISSION = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata (Pre-generated)
TRAIN_META = os.path.join(METADATA_DIR, "train.csv")
VAL_META = os.path.join(METADATA_DIR, "val.csv")
TEST_META = os.path.join(METADATA_DIR, "test.csv")

# Working Directory (Cache & Models)
WORKING_DIR = "./working/idea_12"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission Directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ==========================================
# CACHING PATHS
# ==========================================
# We use .npy for large feature tensors and .parquet for metadata/labels
# to ensure fast I/O during the training phase.

# Training Cache
TRAIN_FEATURES = os.path.join(WORKING_DIR, "train_features.npy")
TRAIN_LABELS_L3 = os.path.join(WORKING_DIR, "train_labels_l3.npy")
TRAIN_IDS = os.path.join(WORKING_DIR, "train_ids.npy")

# Validation Cache
VAL_FEATURES = os.path.join(WORKING_DIR, "val_features.npy")
VAL_LABELS_L3 = os.path.join(WORKING_DIR, "val_labels_l3.npy")
VAL_IDS = os.path.join(WORKING_DIR, "val_ids.npy")

# Test Cache
TEST_FEATURES = os.path.join(WORKING_DIR, "test_features.npy")
TEST_IDS = os.path.join(WORKING_DIR, "test_ids.npy")

# Auxiliary Mappings
HIERARCHY_MAPPING = os.path.join(WORKING_DIR, "hierarchy_mapping.parquet")
CATEGORY_ENCODER = os.path.join(WORKING_DIR, "category_encoder.pkl")

# ==========================================
# DATA PROCESSING HYPERPARAMETERS
# ==========================================
# Image Processing
IMG_SIZE = 224  # Standard input size for ResNet/EfficientNet
BATCH_SIZE_EXTRACT = 256  # Batch size for feature extraction (inference mode)

# Debugging
# Set to an integer (e.g., 50000) to process a subset of data for quick testing.
# Set to None to process the full dataset (Required for final submission).
DEBUG_SAMPLE_SIZE = None

# ==========================================
# MODEL ARCHITECTURE
# ==========================================
# Feature Extractor (Frozen Backbones)
BACKBONE_1 = "resnet50"
BACKBONE_1_DIM = 2048
BACKBONE_2 = "efficientnet_b0"
BACKBONE_2_DIM = 1280

# The input dimension to the MLP is the concatenation of both backbones
INPUT_DIM = BACKBONE_1_DIM + BACKBONE_2_DIM  # 3328

# Hierarchical MLP Head
HIDDEN_DIM = 1024
DROPOUT_RATE = 0.3

# Class Counts (from category_names.csv)
NUM_CLASSES_L1 = 49
NUM_CLASSES_L2 = 483
NUM_CLASSES_L3 = 5270

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
ENSEMBLE_SIZE = 5  # Number of models in the ensemble
BATCH_SIZE_TRAIN = 2048  # Large batch size for MLP training on pre-computed features
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5

# Regularization
MIXUP_ALPHA = 0.2  # Alpha for MixUp regularization on feature vectors

# Model Checkpointing
# Template for saving ensemble members, e.g., model_0.pth, model_1.pth
MODEL_SAVE_PATH_TEMPLATE = os.path.join(WORKING_DIR, "ensemble_model_{}.pth")
