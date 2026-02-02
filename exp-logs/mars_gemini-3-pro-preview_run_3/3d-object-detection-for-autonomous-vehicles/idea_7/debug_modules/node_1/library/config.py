import os
import torch

# -----------------------------------------------------------------------------
# Path Configuration
# -----------------------------------------------------------------------------
DATA_ROOT = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_7"
SUBMISSION_PATH = "./submission/submission.csv"

# Ensure working directory exists
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# -----------------------------------------------------------------------------
# Dataset Configuration
# -----------------------------------------------------------------------------
# Classes identified from data analysis
CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "bicycle",
    "pedestrian",
    "other_vehicle",
    "motorcycle",
    "animal",
    "emergency_vehicle",
]
NUM_CLASSES = len(CLASS_NAMES)

# Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
# Covering a 102.4m x 102.4m area.
# Z range covers typical object heights relative to sensor.
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# Voxelization
# High resolution: 0.08m xy, 4.0m z
# This results in a fine-grained BEV grid.
VOXEL_SIZE = [0.08, 0.08, 4.0]
MAX_POINTS_PER_VOXEL = 32
MAX_NUMBER_OF_VOXELS_TRAIN = 40000
MAX_NUMBER_OF_VOXELS_TEST = 80000

# Grid Size (calculated)
# X: 102.4 / 0.08 = 1280
# Y: 102.4 / 0.08 = 1280
# Z: 8.0 / 4.0 = 2
GRID_SIZE = [1280, 1280, 2]

# Input Features
# x, y, z, intensity
NUM_POINT_FEATURES = 4

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------
# PointPillars Encoder
NUM_FILTERS = [64]  # Features per pillar

# ResNet-FPN Backbone
# Using ResNet18 structure: layers [2, 2, 2, 2]
BACKBONE_LAYERS = [2, 2, 2, 2]
BACKBONE_CHANNELS = [64, 128, 256, 512]
FPN_OUT_CHANNELS = 256

# CenterHead
# Output heads: dict(name=num_channels)
# We use a center-based anchor-free head.
COMMON_HEADS = {
    "reg": 2,  # offset x, y
    "height": 1,  # z center
    "dim": 3,  # l, w, h (log-space)
    "rot": 2,  # sin(r), cos(r)
}
HEAD_CONV = 64  # Channels for head convolution

# -----------------------------------------------------------------------------
# Training Configuration
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Hyperparameters
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
GRAD_NORM_CLIP = 10.0

# Loss Weights
LOSS_WEIGHTS = {"heatmap": 1.0, "reg": 2.0, "height": 1.0, "dim": 2.0, "rot": 1.0}

# Data Loading
NUM_WORKERS = 4
PIN_MEMORY = True

# Debugging / Development
# Set to None to use full dataset, or an integer to limit samples
TRAIN_SUBSET_SIZE = None
VAL_SUBSET_SIZE = None

# -----------------------------------------------------------------------------
# Augmentation Configuration
# -----------------------------------------------------------------------------
# Simple geometric augmentations that preserve consistency
ROTATION_RANGE = [-0.78539816, 0.78539816]  # +/- 45 deg
SCALING_RANGE = [0.95, 1.05]
FLIP_PROB = 0.5
