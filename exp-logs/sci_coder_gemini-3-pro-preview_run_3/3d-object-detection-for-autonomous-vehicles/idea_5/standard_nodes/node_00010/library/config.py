import os
import torch
import numpy as np

# ==============================================================================
# Paths & Directories
# ==============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5"

# Ensure the working directory exists for caching and checkpoints
os.makedirs(WORKING_DIR, exist_ok=True)

# Metadata File Paths
TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

# Submission Output
SUBMISSION_PATH = "./submission/submission.csv"

# ==============================================================================
# Data Configuration
# ==============================================================================
# Class names extracted from dataset analysis
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
# Defined in Ego-Sensor coordinates.
# The Data Loader is responsible for transforming World labels to this frame.
# Range covers 102.4m x 102.4m centered on the ego vehicle.
POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# Voxel Size: [x_size, y_size, z_size]
# 0.1m resolution allows for fine-grained detection of pedestrians.
# z_size = 8.0 collapses the vertical dimension into a single pillar.
VOXEL_SIZE = [0.1, 0.1, 8.0]

# Grid Size calculation: [W, H]
# Result: [1024, 1024]
GRID_SIZE = [
    int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
    int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]),
]

# PointPillars Encoding Settings
MAX_PILLARS = 30000  # Max number of non-empty pillars supported
MAX_POINTS_PER_PILLAR = 32  # Max points sampled per pillar
NUM_POINT_FEATURES = 4  # Features: x, y, z, intensity

# ==============================================================================
# Model Configuration
# ==============================================================================
# Backbone Downsampling Ratio
# Input Grid: 1024x1024 -> Output Map: 256x256
DOWN_RATIO = 4
OUT_SIZE_FACTOR = DOWN_RATIO

# Number of filters in the initial PillarFeatureNet
NUM_FILTERS = 64

# ==============================================================================
# Training Configuration
# ==============================================================================
# Batch size adapted for A100 GPU (40GB)
BATCH_SIZE = 4
NUM_EPOCHS = 20

# Optimization
LEARNING_RATE = 3e-4  # AdamW default start
WEIGHT_DECAY = 0.01
GRAD_NORM_CLIP = 10.0

# Debugging / Iteration Control
# Set to an integer (e.g., 500) to train on a small subset for quick debugging.
# Set to None to train on the full dataset.
DEBUG_N_SAMPLES = None

# ==============================================================================
# Hardware & Reproducibility
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
SEED = 42


def set_deterministic(seed=SEED):
    """
    Sets random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
