import os
import torch

# =============================================================================
# DIRECTORY & PATH CONFIGURATION
# =============================================================================
# Base directories
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_8"
SUBMISSION_DIR = "./submission"

# Ensure output directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Image directories
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test")

# Metadata file paths
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(INPUT_DIR, "sample_submission.csv")

# Cache directory for deterministic data processing
CACHE_DIR = os.path.join(WORKING_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
IMG_SIZE = 320
NUM_CLASSES = 4029  # Total unique classes including 'new_whale'
NUM_WORKERS = 4  # Number of DataLoader workers

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_NAME = "densenet121"
PRETRAINED = True
EMBEDDING_DIM = 512
USE_NECK = True  # Use BN-Neck (Linear -> BN) before ArcFace head

# ArcFace Hyperparameters
MARGIN = 0.50
SCALE = 30.0

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
# Ensemble Seeds: 2 independent models (Cite solution_lesson_node_00027)
SEEDS = [42, 2023]

BATCH_SIZE = 32
EPOCHS = 24  # Increased to ensure convergence (Cite solution_lesson_node_00026)
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
LABEL_SMOOTHING = 0.1

# Learning Rate Scheduler (Cosine Annealing)
SCHEDULER_T_MAX = EPOCHS
SCHEDULER_MIN_LR = 1e-6

# Early Stopping
PATIENCE = 5

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TTA_FLIP = True  # Enable Horizontal Flip Test-Time Augmentation
TOP_K = 5  # Number of predictions per image
