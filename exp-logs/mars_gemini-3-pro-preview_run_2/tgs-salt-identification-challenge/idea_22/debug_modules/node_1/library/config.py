import os
import torch
import random
import numpy as np

# =============================================================================
# Directories and File Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = "./working/idea_22"

# Create cache directory immediately to ensure availability
os.makedirs(CACHE_DIR, exist_ok=True)

# Metadata Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Raw Data Paths
DEPTHS_CSV_PATH = os.path.join(INPUT_DIR, "depths.csv")
TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train", "images")
TRAIN_MASKS_DIR = os.path.join(INPUT_DIR, "train", "masks")
TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test", "images")

# Processed/Cached Data Paths (NPY format)
CACHE_TRAIN_IMAGES = os.path.join(CACHE_DIR, "train_images.npy")
CACHE_TRAIN_MASKS = os.path.join(CACHE_DIR, "train_masks.npy")
CACHE_TRAIN_DEPTHS = os.path.join(CACHE_DIR, "train_depths.npy")
CACHE_TRAIN_IDS = os.path.join(CACHE_DIR, "train_ids.npy")

CACHE_VAL_IMAGES = os.path.join(CACHE_DIR, "val_images.npy")
CACHE_VAL_MASKS = os.path.join(CACHE_DIR, "val_masks.npy")
CACHE_VAL_DEPTHS = os.path.join(CACHE_DIR, "val_depths.npy")
CACHE_VAL_IDS = os.path.join(CACHE_DIR, "val_ids.npy")

CACHE_TEST_IMAGES = os.path.join(CACHE_DIR, "test_images.npy")
CACHE_TEST_DEPTHS = os.path.join(CACHE_DIR, "test_depths.npy")
CACHE_TEST_IDS = os.path.join(CACHE_DIR, "test_ids.npy")

# Model Checkpoints & Submission
TEACHER_CHECKPOINT = os.path.join(WORKING_DIR, "teacher_best.pth")
STUDENT_CHECKPOINT = os.path.join(WORKING_DIR, "student_best.pth")
SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

# =============================================================================
# Data & Model Hyperparameters
# =============================================================================
SEED = 42

# Image Dimensions
ORIG_SIZE = 101
IMG_SIZE = 128  # Padded size for U-Net/ResNet compatibility (divisible by 32)
CHANNELS = 1  # Grayscale (summed RGB)

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS_TEACHER = 50
NUM_EPOCHS_STUDENT = 50
NUM_WORKERS = 4  # Adjusted for available vCPUs

# Compute Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Augmentation Parameters (Elastic Transform & ShiftScaleRotate)
ELASTIC_ALPHA = 120
ELASTIC_SIGMA = 6
ELASTIC_ALPHA_AFFINE = 120 * 0.03
AUG_PROB = 0.5

# Distillation Loss Weights
# Formula: Loss = L_Seg + lambda1 * L_Distill + lambda2 * L_Depth
LAMBDA_SEG = 1.0  # Weight for Ground Truth Segmentation Loss
LAMBDA_DISTILL = 0.5  # Weight for Teacher-Student Distillation Loss
LAMBDA_DEPTH = 0.1  # Weight for Auxiliary Depth Regression Loss


# =============================================================================
# Utility Functions
# =============================================================================
def setup_reproducibility(seed=SEED):
    """
    Sets fixed random seeds for Python, NumPy, and PyTorch to ensure
    reproducible results across runs.

    Args:
        seed (int): The seed value to use. Defaults to config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
