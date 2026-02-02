import os
import torch

# =============================================================================
# General Configuration
# =============================================================================
SEED = 42
NUM_WORKERS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Directories and Paths
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
SUBMISSION_DIR = "./submission"

# Create working and submission directories if they don't exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Auxiliary Data
MEGADETECTOR_PATH = os.path.join(INPUT_DIR, "iwildcam2020_megadetector_results.json")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_DIR, "sample_submission.csv")

# Output Paths
SUBMISSION_FILE_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
BBOX_CACHE_PATH = os.path.join(WORKING_DIR, "bbox_cache.parquet")
BEST_MODEL_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# =============================================================================
# Data Processing Parameters
# =============================================================================
IMAGE_SIZE = 384
CROP_MARGIN = 0.2  # 20% margin around the bounding box
NUM_CLASSES = 185  # Number of unique classes in the training set

# =============================================================================
# Model Parameters
# =============================================================================
MODEL_NAME = "convnext_small.fb_in1k"
PRETRAINED = True

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

# Optimizer & Scheduler
USE_COSINE_SCHEDULER = True
WARMUP_EPOCHS = 1

# Loss Function
USE_FOCAL_LOSS = True
FOCAL_LOSS_GAMMA = 2.0

# Early Stopping
# Set patience equal to EPOCHS to effectively disable it while keeping the logic hook
EARLY_STOPPING_PATIENCE = 15

# =============================================================================
# Inference Parameters
# =============================================================================
USE_TTA = True  # Test Time Augmentation (Horizontal Flip)
