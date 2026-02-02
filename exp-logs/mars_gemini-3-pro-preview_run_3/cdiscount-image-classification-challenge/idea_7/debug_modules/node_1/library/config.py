import os
import torch
import numpy as np
import random

# =============================================================================
# DIRECTORY SETUP
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_7"

# Ensure the working directory exists for caching features and models
os.makedirs(WORKING_DIR, exist_ok=True)

# =============================================================================
# FILE PATHS
# =============================================================================
# Raw BSON Data
TRAIN_BSON_PATH = os.path.join(INPUT_DIR, "train.bson")
TEST_BSON_PATH = os.path.join(INPUT_DIR, "test.bson")

# Auxiliary Data
CATEGORY_NAMES_PATH = os.path.join(INPUT_DIR, "category_names.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Pre-computed Metadata (Indices)
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Cache Paths for Decoupled Features (Numpy format)
# These store the 1280-dim embeddings extracted from EfficientNet-B0
TRAIN_FEATURES_PATH = os.path.join(WORKING_DIR, "train_features.npy")
TRAIN_LABELS_PATH = os.path.join(WORKING_DIR, "train_labels.npy")
VAL_FEATURES_PATH = os.path.join(WORKING_DIR, "val_features.npy")
VAL_LABELS_PATH = os.path.join(WORKING_DIR, "val_labels.npy")
TEST_FEATURES_PATH = os.path.join(WORKING_DIR, "test_features.npy")
TEST_IDS_PATH = os.path.join(WORKING_DIR, "test_ids.npy")

# Model Checkpoints and Output
MODEL_PATH = os.path.join(WORKING_DIR, "hierarchical_effnet_b0.pth")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# =============================================================================
# HYPERPARAMETERS & CONSTANTS
# =============================================================================
# Reproducibility
SEED = 42

# Image Preprocessing (EfficientNet Defaults)
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Feature Extraction
# EfficientNet-B0 outputs a 1280-dimensional vector before the classifier
EMBEDDING_DIM = 1280
BATCH_SIZE_EXTRACT = 256  # Batch size for image forward pass (GPU memory dependent)

# Tabular Training (Hierarchical Classifier)
# Large batch size for stable gradients on decoupled features
BATCH_SIZE_TRAIN = 4096
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
PATIENCE = 5  # Early stopping patience

# Hardware
# Using 12 vCPUs for data loading and A100 for computation
NUM_WORKERS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def seed_everything(seed=SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # False ensures strict reproducibility
