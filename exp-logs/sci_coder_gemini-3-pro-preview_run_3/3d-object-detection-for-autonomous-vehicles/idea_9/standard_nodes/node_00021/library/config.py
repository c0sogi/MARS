import os
import torch
import numpy as np


class Config:
    """
    Global configuration for the Two-Stage PointPillars 3D Object Detection pipeline.
    """

    # ==============================================================================
    # PATHS & DIRECTORIES
    # ==============================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_9"

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Metadata file paths
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Roots (for coordinate transformation access)
    TRAIN_DATA_ROOT = os.path.join(INPUT_DIR, "train_data")
    TEST_DATA_ROOT = os.path.join(INPUT_DIR, "test_data")

    # Cache Directories
    GT_DATABASE_DIR = os.path.join(WORKING_DIR, "gt_database")
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Submission
    SUBMISSION_PATH = "./submission/submission.csv"
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==============================================================================
    # HARDWARE & REPRODUCIBILITY
    # ==============================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Number of dataloader workers
    SEED = 42

    # ==============================================================================
    # DATASET SPECIFICATIONS
    # ==============================================================================
    # Class Names based on dataset analysis
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

    # Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
    # Using a wide range to capture objects in the scene
    POINT_CLOUD_RANGE = [-100.0, -100.0, -5.0, 100.0, 100.0, 3.0]

    # Input Features: x, y, z, intensity
    NUM_POINT_FEATURES = 4

    # ==============================================================================
    # VOXELIZATION (POINTPILLARS)
    # ==============================================================================
    # Voxel Size: [x, y, z]
    # z=8.0 ensures the entire z-axis is collapsed into a single pillar
    VOXEL_SIZE = [0.5, 0.5, 8.0]

    # Max points per voxel/pillar
    MAX_POINTS_PER_VOXEL = 32

    # Max number of voxels (pillars) to generate
    MAX_VOXELS_TRAIN = 16000
    MAX_VOXELS_TEST = 40000

    # Calculate Grid Size
    GRID_SIZE = [
        int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
        int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]),
        int((POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) / VOXEL_SIZE[2]),
    ]  # [400, 400, 1]

    # ==============================================================================
    # MODEL ARCHITECTURE
    # ==============================================================================
    # Pillar Feature Net
    HIDDEN_DIM = 64

    # Backbone (ResNet-like FPN)
    LAYER_STRIDES = [1, 2, 2]
    LAYER_NUMS = [3, 5, 5]
    UP_STRIDES = [1, 2, 4]
    NUM_FILTERS = [64, 128, 256]

    # Stage 1: CenterHead
    # Downsampling ratio of the backbone output relative to the input grid
    # If backbone output is 200x200 for 800x800 input, ratio is 4
    # With the above settings, it's likely 2 or 4 depending on implementation.
    # We assume a standard stride of 2 for the final feature map.
    FEATURE_MAP_STRIDE = 2

    # Stage 2: RoI Head & Rectification
    ROI_SIZE = 7  # RoI Align output size (7x7)
    ROI_OUT_CHANNELS = 256  # Dimension after RoI pooling

    # IoU Rectification Hyperparameter
    # Score_final = Score_cls * (IoU_pred ^ alpha)
    IOU_RECT_ALPHA = 0.5

    # ==============================================================================
    # TRAINING HYPERPARAMETERS
    # ==============================================================================
    BATCH_SIZE = 4  # Conservative batch size for A100 with complex model
    EPOCHS = 15

    # Optimizer (AdamW)
    LR = 0.003
    WEIGHT_DECAY = 0.01

    # Gradient Clipping
    GRAD_CLIP_NORM = 10.0

    # Loss Weights
    LOSS_WEIGHTS = {
        "cls_weight": 1.0,
        "loc_weight": 2.0,
        "iou_weight": 1.0,  # Weight for the Stage 2 rectification branch
    }

    # ==============================================================================
    # DATA AUGMENTATION (GT SAMPLING)
    # ==============================================================================
    USE_GT_AUGMENTATION = True

    # Number of objects to sample per class per frame
    DB_SAMPLER = {
        "car": 15,
        "truck": 10,
        "bus": 10,
        "other_vehicle": 10,
        "pedestrian": 10,
        "bicycle": 10,
        "motorcycle": 10,
        "animal": 5,
        "emergency_vehicle": 5,
    }

    # Minimum points required in a GT database entry to be considered valid
    MIN_POINTS_IN_GT = 5

    # ==============================================================================
    # DEBUGGING & SUBSETS
    # ==============================================================================
    # If True, runs on a small subset of data for rapid prototyping
    DEBUG = False
    SUBSET_SIZE = 500  # Number of samples to use if DEBUG is True

    @staticmethod
    def print_config():
        print("\n" + "=" * 40)
        print("CONFIG SETTINGS")
        print("=" * 40)
        for key, val in Config.__dict__.items():
            if not key.startswith("__") and not callable(val):
                print(f"{key:<25}: {val}")
        print("=" * 40 + "\n")
