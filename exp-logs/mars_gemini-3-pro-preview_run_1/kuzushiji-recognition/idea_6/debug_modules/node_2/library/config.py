import os
import torch

# =============================================================================
# PATHS & DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
TRAIN_IMGS_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMGS_DIR = os.path.join(INPUT_DIR, "test_images")
UNICODE_MAP_PATH = os.path.join(INPUT_DIR, "unicode_translation.csv")

METADATA_DIR = "./metadata"
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Working directory for caching intermediate files and saving models
WORKING_DIR = "./working/idea_6"
os.makedirs(WORKING_DIR, exist_ok=True)

# Submission directory
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
BACKBONE = "convnext_tiny"  # Selected for balance of receptive field and speed
IMG_SIZE = 1024  # Input resolution (square)
IN_CHANNELS = 3
# Number of classes based on unicode_translation.csv (4782 lines)
# This covers the full potential vocabulary of the dataset.
NUM_CLASSES = 4782
DOWN_RATIO = 4  # Stride for the output heatmap (Input / 4)

# =============================================================================
# DATA & PREPROCESSING
# =============================================================================
# Standard ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
BATCH_SIZE = 4  # Reduced to prevent OOM on dense gradients (Cite debug_lesson_13)
NUM_EPOCHS = 35  # Sufficient for convergence with 'tiny' backbone
LEARNING_RATE = 2e-4  # AdamW base LR
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# INFERENCE HYPERPARAMETERS
# =============================================================================
MAX_DETECTIONS = 1200  # Constraint per page
CONF_THRESHOLD = 0.1  # Minimum heatmap confidence to extract a point

# =============================================================================
# DEBUGGING
# =============================================================================
# If True, runs on a small subset of data for quick pipeline verification
DEBUG = False
