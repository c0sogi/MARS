import os
import torch
import pandas as pd
import numpy as np

# ==========================================
# Global Random Seed
# ==========================================
SEED = 42

# ==========================================
# File Paths and Directories
# ==========================================
INPUT_ROOT = "./input"
METADATA_DIR = "./metadata"
# Working directory for caching and checkpoints
WORKING_DIR = "./working/idea_10"
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata files
TRAIN_CSV = os.path.join(METADATA_DIR, "train.csv")
VAL_CSV = os.path.join(METADATA_DIR, "val.csv")
TEST_CSV = os.path.join(METADATA_DIR, "test.csv")

# Output paths
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
SUBMISSION_PATH = "./submission/submission.csv"

# ==========================================
# Data Parameters
# ==========================================
# Original image size
ORIG_SIZE = 101
# Input size for the network (padded to be divisible by 32)
IMG_SIZE = 128
# Number of channels (1 for grayscale seismic, but model may adapt)
IN_CHANNELS = 1

# Data Loading
BATCH_SIZE = 32
NUM_WORKERS = 4

# Debugging / Development
# Set to True to train on a small subset for quick pipeline verification
DEBUG = False
DEBUG_SAMPLE_SIZE = 50

# ==========================================
# Model Parameters
# ==========================================
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# Training Hyperparameters
# ==========================================
EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
# Cosine Annealing T_max (usually equal to epochs)
SCHEDULER_T_MAX = EPOCHS

# ==========================================
# Augmentation Parameters
# ==========================================
AUG_PROB = 0.2
# Elastic Transform parameters
ELASTIC_ALPHA = 120
ELASTIC_SIGMA = 6

# ==========================================
# Normalization Constants
# ==========================================
# Standard ImageNet Mean and Std (RGB)
# Even for grayscale, we often project to these stats if using pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Depth Normalization Statistics
# Calculated dynamically from the training metadata to ensure accuracy.
# These are used to normalize the 'z' input before the depth injection module.
DEPTH_MEAN = 0.0
DEPTH_STD = 1.0

if os.path.exists(TRAIN_CSV):
    try:
        _df = pd.read_csv(TRAIN_CSV)
        if "z" in _df.columns:
            DEPTH_MEAN = float(_df["z"].mean())
            DEPTH_STD = float(_df["z"].std())
        del _df
    except Exception as e:
        print(
            f"Warning: Could not calculate depth stats from {TRAIN_CSV}. Using defaults. Error: {e}"
        )

# Value to use for depth when it is missing (e.g., during inference if depth is dropped)
# Using 0.0 assumes the input is standardized (mean centered)
DEPTH_FILL_VALUE = 0.0
