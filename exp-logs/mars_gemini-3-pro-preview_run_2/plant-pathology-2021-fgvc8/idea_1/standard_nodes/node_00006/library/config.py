import os
import torch

# ==========================================
# Global Configuration & Hyperparameters
# ==========================================

# Random Seed for Reproducibility
SEED = 42

# ==========================================
# Data Directories & Paths
# ==========================================

# Input Directories
INPUT_DIR = "./input"
TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")

# Metadata Paths (Pre-generated)
METADATA_DIR = "./metadata"
TRAIN_META_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_META_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_META_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Directories
WORKING_DIR = "./working"
IDEA_DIR = os.path.join(WORKING_DIR, "idea_1")  # For caching intermediate data
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Create necessary output directories
os.makedirs(IDEA_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# ==========================================
# Model & Training Configuration
# ==========================================

# Model Architecture
# Using a medium EfficientNetV2 variant for better capacity
MODEL_NAME = "tf_efficientnetv2_m.in21k_ft_in1k"

# Input Image Dimensions
# Reduced to 384x384 to prevent OOM
IMG_SIZE = 384

# Training Hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3

# ==========================================
# Target Classes
# ==========================================

# Sorted alphabetically to ensure consistent indexing
CLASSES = ["complex", "frog_eye_leaf_spot", "healthy", "powdery_mildew", "rust", "scab"]

NUM_CLASSES = len(CLASSES)

# ==========================================
# Compute Configuration
# ==========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of workers for data loading
# Using available CPUs (12 vCPUs available in env)
NUM_WORKERS = os.cpu_count()
