import os
import torch
import numpy as np
import random

# ==============================================================================
# PATHS & DIRECTORIES
# ==============================================================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
CACHE_DIR = os.path.join(WORKING_DIR, "idea_1")
SUBMISSION_DIR = "./submission"
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Metadata Files
TRAIN_METADATA = os.path.join(METADATA_DIR, "train.csv")
VAL_METADATA = os.path.join(METADATA_DIR, "val.csv")
TEST_METADATA = os.path.join(METADATA_DIR, "test.csv")

# JSON Data Directories (needed for calibration/pose loading)
TRAIN_DATA_JSON_DIR = os.path.join(INPUT_DIR, "train_data")
TEST_DATA_JSON_DIR = os.path.join(INPUT_DIR, "test_data")

# ==============================================================================
# DATASET CONFIGURATION
# ==============================================================================
# Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
# Covering +/- 100m in X/Y and -5m to +3m in Z (Ego Frame)
POINT_CLOUD_RANGE = [-100.0, -100.0, -5.0, 100.0, 100.0, 3.0]

# Voxel Size for BEV Rasterization (meters)
# A voxel size of 0.2m results in a 1000x1000 grid for a 200m range
VOXEL_SIZE = [0.2, 0.2]  # [x_step, y_step]

# Calculated Grid Size (W, H)
GRID_SIZE = [
    int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
    int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]),
]

# Input Channels for the BEV Map
# 1. Max Height
# 2. Mean Intensity
# 3. Point Density
BEV_CHANNELS = 3

# Class Mapping (based on EDA)
CLASS_NAMES = [
    "car",
    "other_vehicle",
    "pedestrian",
    "bicycle",
    "truck",
    "bus",
    "motorcycle",
    "animal",
    "emergency_vehicle",
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================
BACKBONE = "resnet18"  # Lightweight backbone
PRETRAINED = True
HEAD_CONV = 64  # Channels in the detection head

# Downsampling factor of the backbone (ResNet usually 32, but we might modify or use upsampling)
# For a simple CenterNet style, we often want output stride 4.
# If using standard ResNet, we might need Deconvolutions.
# For this config, we assume the model handles stride internally.
DOWN_RATIO = 4

# ==============================================================================
# TRAINING CONFIGURATION
# ==============================================================================
SEED = 42
BATCH_SIZE = 16  # Adjusted for A100 40GB (can likely go higher, but 16 is safe)
NUM_EPOCHS = 25  # Extended training duration for regression convergence (Cite solution_lesson_node_00009)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4  # 12 vCPUs available

# Early Stopping
PATIENCE = 5
MIN_DELTA = 0.001

# Loss Weights (Cite solution_lesson_node_00007)
HM_WEIGHT = 1.0  # Heatmap Loss
WH_WEIGHT = 1.0  # Width/Length/Height Loss
OFF_WEIGHT = 1.0  # Local Offset Loss (if used)
Z_WEIGHT = 1.0  # Z-coordinate Loss
ROT_WEIGHT = 1.0  # Rotation/Yaw Loss


# ==============================================================================
# UTILITIES
# ==============================================================================
def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Returns the appropriate device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
