import os
import torch

# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"

# Ensure the working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test.csv")

# Output Paths
SUBMISSION_DIR = "./submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")
MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "best_model.pth")

# =============================================================================
# DATA PREPROCESSING
# =============================================================================
# Image Dimensions
IMG_HEIGHT = 512
IMG_WIDTH = 512
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Dataloader Settings
# A100 40GB can handle larger batches, but 768x768 is memory intensive.
BATCH_SIZE = 8
NUM_WORKERS = 12

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
BACKBONE = "tf_efficientnet_b2"
PRETRAINED = True

# Input Channels: 3
# Channel 0: Mammogram Image
# Channel 1: Spatially Broadcasted Age (Standard Scaled)
# Channel 2: Spatially Broadcasted Implant Status
IN_CHANNELS = 3
NUM_CLASSES = 1

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================
SEED = 42
EPOCHS = 8
LR = 1e-4
WEIGHT_DECAY = 0.01

# Loss Function Weights
# Based on imbalance ratio ~1:47
POS_WEIGHT = 47.0

# Stochastic Modality Dropout
# Probability of zeroing out Age/Implant channels during training
MODALITY_DROPOUT_PROB = 0.5

# Gradient Strategy
# Explicitly disabled as per Lesson 00009
USE_GRADIENT_CLIPPING = False

# =============================================================================
# COMPUTE
# =============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
