import os
import numpy as np

# ==============================================================================
# PATHS AND DIRECTORIES
# ==============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_3"
SUBMISSION_PATH = "./submission/submission.csv"

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

# ==============================================================================
# DATASET SPECIFICATIONS
# ==============================================================================
# Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
# This covers a 102.4m x 102.4m area centered at (0,0) with Z covering vehicle heights
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# Voxel Size: [x_size, y_size, z_size]
# Z size is set to the full range height (8.0m) to create pillars (no z-axis voxelization)
VOXEL_SIZE = [0.16, 0.16, 8.0]

# Calculate Grid Size: [W, H, D] -> [640, 640, 1]
GRID_SIZE = [
    int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
    int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]),
    1,
]

# Point Sampling Configuration
MAX_POINTS_PER_PILLAR = 32
MAX_PILLARS_TRAIN = 16000
MAX_PILLARS_TEST = 40000

# Feature Dimensions
NUM_POINT_FEATURES = 4  # x, y, z, intensity
NUM_PILLAR_FEATURES = 64  # Output channels of the PillarFeatureNet

# ==============================================================================
# CLASS AND ANCHOR CONFIGURATION
# ==============================================================================
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

CLASS_TO_ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i + 1: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Anchor Configuration
# Sizes are [width, length, height] derived from dataset analysis
# Rotations are in radians [0, pi/2]
# Bottom heights are relative to sensor origin (approx ground level at -1.78m for vehicles)
ANCHOR_CONFIGS = [
    {
        "class_name": "car",
        "anchor_sizes": [[1.93, 4.75, 1.72]],
        "anchor_rotations": [0, 1.57],
        "anchor_bottom_heights": [-1.78],
        "matched_threshold": 0.6,
        "unmatched_threshold": 0.45,
    },
    {
        "class_name": "truck",
        "anchor_sizes": [[2.82, 10.36, 3.46]],
        "anchor_rotations": [0, 1.57],
        "anchor_bottom_heights": [-1.78],
        "matched_threshold": 0.55,
        "unmatched_threshold": 0.4,
    },
    {
        "class_name": "bus",
        "anchor_sizes": [[2.95, 12.38, 3.42]],
        "anchor_rotations": [0, 1.57],
        "anchor_bottom_heights": [-1.78],
        "matched_threshold": 0.55,
        "unmatched_threshold": 0.4,
    },
    {
        "class_name": "bicycle",
        "anchor_sizes": [[0.64, 1.75, 1.46]],
        "anchor_rotations": [0, 1.57],
        "anchor_bottom_heights": [-0.6],
        "matched_threshold": 0.5,
        "unmatched_threshold": 0.35,
    },
    {
        "class_name": "pedestrian",
        "anchor_sizes": [[0.77, 0.81, 1.77]],
        "anchor_rotations": [0, 1.57],
        "anchor_bottom_heights": [-0.6],
        "matched_threshold": 0.5,
        "unmatched_threshold": 0.35,
    },
    {
        "class_name": "other_vehicle",
        "anchor_sizes": [[2.80, 8.18, 3.24]],
        "anchor_rotations": [0, 1.57],
        "anchor_bottom_heights": [-1.78],
        "matched_threshold": 0.5,
        "unmatched_threshold": 0.35,
    },
    {
        "class_name": "motorcycle",
        "anchor_sizes": [[1.00, 2.40, 1.52]],
        "anchor_rotations": [0, 1.57],
        "anchor_bottom_heights": [-0.6],
        "matched_threshold": 0.5,
        "unmatched_threshold": 0.3,
    },
    {
        "class_name": "animal",
        "anchor_sizes": [[0.38, 0.77, 0.57]],
        "anchor_rotations": [0, 1.57],
        "anchor_bottom_heights": [-0.6],
        "matched_threshold": 0.5,
        "unmatched_threshold": 0.3,
    },
    {
        "class_name": "emergency_vehicle",
        "anchor_sizes": [[2.89, 7.76, 2.97]],
        "anchor_rotations": [0, 1.57],
        "anchor_bottom_heights": [-1.78],
        "matched_threshold": 0.5,
        "unmatched_threshold": 0.35,
    },
]

# ==============================================================================
# MODEL ARCHITECTURE (RPN BACKBONE)
# ==============================================================================
LAYER_STRIDES = [1, 2, 2]
LAYER_FILTERS = [64, 128, 256]
UPSAMPLE_STRIDES = [1, 2, 4]
NUM_UPSAMPLE_FILTERS = [128, 128, 128]

# ==============================================================================
# TRAINING HYPERPARAMETERS
# ==============================================================================
SEED = 42
BATCH_SIZE = 4  # Adjusted for A100 (40GB)
NUM_WORKERS = 4
EPOCHS = 20
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.01
GRAD_NORM_CLIP = 10.0
WARMUP_EPOCHS = 1

# Loss Weights
LOSS_WEIGHTS = {"cls_weight": 1.0, "box_weight": 2.0, "dir_weight": 0.2}

# ==============================================================================
# INFERENCE & POST-PROCESSING
# ==============================================================================
SCORE_THRESHOLD = 0.1
NMS_IOU_THRESHOLD = 0.1
MAX_DETECTIONS = 500
