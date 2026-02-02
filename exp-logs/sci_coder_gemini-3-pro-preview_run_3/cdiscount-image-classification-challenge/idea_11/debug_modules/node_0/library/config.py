import os
import torch

# ==========================================
# DIRECTORY CONFIGURATION
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Specific cache directory for this experiment (Idea 11)
CACHE_DIR = os.path.join(WORKING_DIR, "idea_11")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# FILE PATHS
# ==========================================
# Raw BSON files
TRAIN_BSON_PATH = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON_PATH = os.path.join(INPUT_DIR, "test.bson")
TRAIN_EXAMPLE_BSON_PATH = os.path.join(INPUT_DIR, "train_example.bson")

# Metadata files (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Auxiliary files
CATEGORY_NAMES_PATH = os.path.join(INPUT_DIR, "category_names.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Cached Feature Paths (Decoupled storage)
# These will be generated/loaded by the data processing module
TRAIN_FEATURES_PATH = os.path.join(CACHE_DIR, "train_features.npy")
TRAIN_LABELS_PATH = os.path.join(CACHE_DIR, "train_labels.npy")
VAL_FEATURES_PATH = os.path.join(CACHE_DIR, "val_features.npy")
VAL_LABELS_PATH = os.path.join(CACHE_DIR, "val_labels.npy")
TEST_FEATURES_PATH = os.path.join(CACHE_DIR, "test_features.npy")
TEST_IDS_PATH = os.path.join(CACHE_DIR, "test_ids.npy")

# Model Checkpoint
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "hierarchical_cascade_model.pth")

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
# Input dimension: ResNet50 (2048) + EfficientNet-B0 (1280) = 3328
INPUT_DIM = 3328

# Hierarchical Class Counts
NUM_CLASSES_L1 = 49
NUM_CLASSES_L2 = 483
NUM_CLASSES_L3 = 5270

# Architecture settings
HIDDEN_DIM = 1024
DROPOUT_RATE = 0.2

# ==========================================
# TRAINING HYPERPARAMETERS
# ==========================================
SEED = 42
BATCH_SIZE = 2048  # Large batch size for feature-based training
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

# MixUp Augmentation
USE_MIXUP = True
MIXUP_ALPHA = 0.2

# ==========================================
# COMPUTE RESOURCES
# ==========================================
NUM_WORKERS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# DATA PROCESSING CONFIG
# ==========================================
# Image size for backbones (standard)
IMG_SIZE = 224

# Batch size for feature extraction (inference mode)
EXTRACTION_BATCH_SIZE = 256
