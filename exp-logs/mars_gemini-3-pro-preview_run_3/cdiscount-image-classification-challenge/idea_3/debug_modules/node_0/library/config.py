import os
import torch

# ==========================================
# DIRECTORIES
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_3")
SUBMISSION_DIR = "./submission"

# Create necessary directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# FILE PATHS
# ==========================================
# Raw Data
TRAIN_BSON_PATH = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON_PATH = os.path.join(INPUT_DIR, "test.bson")
CATEGORY_NAMES_PATH = os.path.join(INPUT_DIR, "category_names.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Metadata (Pre-generated)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(CACHE_DIR, "best_model.pth")

# Cache Files (Features & Labels)
# We use .npy for efficient loading of large arrays
TRAIN_FEATURES_PATH = os.path.join(CACHE_DIR, "train_features.npy")
TRAIN_LABELS_PATH = os.path.join(CACHE_DIR, "train_labels.npy")
VAL_FEATURES_PATH = os.path.join(CACHE_DIR, "val_features.npy")
VAL_LABELS_PATH = os.path.join(CACHE_DIR, "val_labels.npy")
TEST_FEATURES_PATH = os.path.join(CACHE_DIR, "test_features.npy")
TEST_IDS_PATH = os.path.join(CACHE_DIR, "test_ids.npy")
CLASS_WEIGHTS_PATH = os.path.join(CACHE_DIR, "class_weights.npy")

# ==========================================
# MODEL HYPERPARAMETERS
# ==========================================
# Backbone: ResNet-50
FEATURE_DIM = 2048
MAX_IMAGES_PER_PRODUCT = 4

# Architecture details
HIDDEN_DIM = 1024
DROPOUT_RATE = 0.5
NUM_CLASSES = 5270  # Based on category_names.csv and EDA

# ==========================================
# TRAINING SETTINGS
# ==========================================
SEED = 42
BATCH_SIZE = (
    4096  # Large batch size since we are training on pre-computed features in RAM
)
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # Early stopping patience

# ==========================================
# HARDWARE
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 12  # Available vCPUs
PIN_MEMORY = True

# ==========================================
# DATA LOADING
# ==========================================
# Image preprocessing settings matching ResNet-50 requirements
IMG_SIZE = 224
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
