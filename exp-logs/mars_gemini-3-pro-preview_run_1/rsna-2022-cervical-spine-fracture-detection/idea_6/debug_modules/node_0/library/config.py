import os
import torch

# ====================================================
# GENERAL CONFIGURATION
# ====================================================
# Random seed for reproducibility
SEED = 42

# Debug mode: Set to True to use a subset of data for quick testing
DEBUG = False

# Compute environment
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Number of workers for data loading (adjust based on available vCPUs)
NUM_WORKERS = 12

# ====================================================
# DIRECTORY PATHS
# ====================================================
# Read-only input directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"

# Working directories for artifacts
WORKING_DIR = "./working/idea_6"
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure writable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Dataset specific paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

TRAIN_IMAGES_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(INPUT_DIR, "test_images")
SEGMENTATION_DIR = os.path.join(INPUT_DIR, "segmentations")
TRAIN_BBOXES_PATH = os.path.join(INPUT_DIR, "train_bounding_boxes.csv")

# ====================================================
# DATA PREPROCESSING CONSTANTS
# ====================================================
# DICOM Windowing (Bone Window)
BONE_WINDOW_CENTER = 400
BONE_WINDOW_WIDTH = 1800

# Image Dimensions
# Full slice size for Stage 1 Localizer
FULL_IMAGE_SIZE = 512
# High-res crop size for Stage 2 Encoder
CROP_IMAGE_SIZE = 256

# Input Channels
# Stage 2 uses 2.5D approach: 3 slices (RGB) + 1 Mask (Alpha)
ENCODER_IN_CHANNELS = 4

# Anatomical Definitions
VERTEBRAE_CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
NUM_VERTEBRAE = len(VERTEBRAE_CLASSES)
# Segmentation classes: 0 (Background) + 7 Vertebrae
NUM_SEG_CLASSES = NUM_VERTEBRAE + 1

# ====================================================
# TRAINING HYPERPARAMETERS
# ====================================================

# Global Batch Size (can be overridden by stage configs)
BATCH_SIZE = 16

# ----------------------------------------------------
# Stage 1: Multi-Class Anatomical Localizer (2D U-Net)
# ----------------------------------------------------
STAGE1_CONFIG = {
    "model_name": "unet_efficientnet_b0",
    "batch_size": 16,
    "epochs": 10 if not DEBUG else 1,
    "lr": 1e-4,
    "image_size": FULL_IMAGE_SIZE,
    "num_classes": NUM_SEG_CLASSES,
}

# ----------------------------------------------------
# Stage 2: Mask-Conditioned Feature Encoder (2.5D CNN)
# ----------------------------------------------------
STAGE2_CONFIG = {
    "backbone": "tf_efficientnetv2_s",
    "batch_size": 32,
    "epochs": 5 if not DEBUG else 1,
    "lr": 3e-4,
    "image_size": CROP_IMAGE_SIZE,
    "in_channels": ENCODER_IN_CHANNELS,
    "feature_dim": 1280,  # Output dimension of the backbone
}

# ----------------------------------------------------
# Stage 3: Anatomically-Grouped Recurrent Aggregator (Bi-GRU)
# ----------------------------------------------------
STAGE3_CONFIG = {
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 4,  # Patient-level batch size (sequence of features)
    "epochs": 10 if not DEBUG else 1,
    "lr": 5e-4,
    "max_seq_length": 300,  # Cap sequence length for memory stability
    "input_dim": STAGE2_CONFIG["feature_dim"]
    + NUM_VERTEBRAE,  # Features + One-hot Anatomy ID
}

# Aggregated Learning Rates for easy reference
LEARNING_RATES = {
    "stage1": STAGE1_CONFIG["lr"],
    "stage2": STAGE2_CONFIG["lr"],
    "stage3": STAGE3_CONFIG["lr"],
}
