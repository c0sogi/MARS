import os
import torch

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
# Root directory for read-only input data
INPUT_DIR = "./input"

# Directory containing generated metadata (train.csv, val.csv, test.csv)
METADATA_DIR = "./metadata"

# Root directory for working files (outputs, caches, models)
WORKING_DIR = "./working"

# Specific cache directory for this solution strategy (Idea 37)
# Used for saving processed datasets, pseudo-labels, and checkpoints
CACHE_DIR = os.path.join(WORKING_DIR, "idea_37")
CHECKPOINT_DIR = os.path.join(CACHE_DIR, "checkpoints")
SUBMISSION_DIR = "./submission"

# Ensure essential writable directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# =============================================================================
# DATA SPECIFICATIONS
# =============================================================================
# Original image dimensions
ORIG_HEIGHT = 101
ORIG_WIDTH = 101

# Target dimensions for training/inference (padded to be divisible by 32)
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Input channels
# 1 channel (Grayscale) is standard, though ResNet backbone expects 3.
# The model definition will handle the conversion (summing weights or expanding input).
IN_CHANNELS = 1

# Number of folds for Cross-Validation
N_FOLDS = 5

# Random Seed for reproducibility
SEED = 42

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
# Backbone architecture
BACKBONE = "resnet34"
PRETRAINED = True

# Decoder channels (Wide-LinkNet specification: in_channels // 4)
# This is handled in the model logic, but we note the architecture type here.
DECODER_TYPE = "Wide-LinkNet"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
# Hardware settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4  # Safe for 12 vCPUs

# Optimization
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2  # Standard for AdamW

# Training Schedules
# Stage 1: Specialist Teacher Training (Supervised)
STAGE1_EPOCHS = 50
# Gating threshold: Discard folds with validation mAP < 0.75
TEACHER_GATING_THRESHOLD = 0.75

# Stage 3: Student Distillation (Semi-Supervised / Multi-Task)
STAGE3_EPOCHS = 50

# =============================================================================
# STRATEGY SPECIFICS: MARGINALIZED DISTILLATION
# =============================================================================
# Depth Scan Values (in Standard Deviations)
# Used in Stage 2 to generate marginalized soft pseudo-labels for the test set
Z_SCAN_VALUES = [-1.5, -0.75, 0.0, 0.75, 1.5]

# Augmentation Parameters
# Non-Rigid: Elastic Transform
AUG_ELASTIC_ALPHA = 120
AUG_ELASTIC_SIGMA = 6
AUG_ELASTIC_PROB = 0.2

# Rigid: ShiftScaleRotate
AUG_RIGID_PROB = 0.2

# =============================================================================
# INFERENCE & SUBMISSION
# =============================================================================
# IoU Thresholds for metric calculation
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# Path for the final submission file
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
