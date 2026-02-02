import os
import torch

# ------------------------------------------------------------------------------
# Global Paths
# ------------------------------------------------------------------------------
INPUT_DIR = "./input"
TRAIN_DIR = os.path.join(INPUT_DIR, "train")
TEST_DIR = os.path.join(INPUT_DIR, "test")
METADATA_DIR = "./metadata"

# Working directory for Idea 25 (Caching and Models)
WORKING_DIR = "./working/idea_25"
os.makedirs(WORKING_DIR, exist_ok=True)

# Output Submission
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
SEED = 42

# ------------------------------------------------------------------------------
# Data Processing Hyperparameters
# ------------------------------------------------------------------------------
# Image Dimensions
IMG_SIZE = 224

# Volumetric Stacking
# We use 4 modalities (FLAIR, T1w, T1wCE, T2w)
# We extract 3 slices per modality using a fixed stride (Cite solution_lesson_node_00037)
# This ensures geometric alignment with pre-trained 3-channel filters (Cite solution_lesson_node_00068)
NUM_SLICES = 3
STRIDE = [-5, 0, 5]
TOTAL_CHANNELS = 4 * NUM_SLICES  # 12 channels total

# ROI Selection (FLAIR Integral)
ROI_DEPTH_MIN = 0.15
ROI_DEPTH_MAX = 0.85

# Augmentation
ROTATION_DEGREES = 15  # +/- 15 degrees

# ------------------------------------------------------------------------------
# Model Hyperparameters
# ------------------------------------------------------------------------------
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
DROPOUT_RATE = 0.3  # For the regularized head

# ------------------------------------------------------------------------------
# Training Hyperparameters
# ------------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 5  # Early stopping patience

# Compute
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
