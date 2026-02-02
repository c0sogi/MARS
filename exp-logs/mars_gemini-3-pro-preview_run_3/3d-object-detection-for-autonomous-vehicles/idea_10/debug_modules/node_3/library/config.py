import os
import torch
import numpy as np


class Config:
    # -------------------------------------------------------------------------
    # 1. Paths & Directories
    # -------------------------------------------------------------------------
    DATA_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_10"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    GT_DATABASE_DIR = os.path.join(WORKING_DIR, "gt_database")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(GT_DATABASE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Hardware & Reproducibility
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    SEED = 42

    # -------------------------------------------------------------------------
    # 3. Voxelization & Grid Configuration
    # -------------------------------------------------------------------------
    # High-resolution voxel size for precise IoU matching
    VOXEL_SIZE = [0.2, 0.2, 4.0]

    # Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
    # Range dimensions: 153.6m x 153.6m x 4.0m
    # Grid dimensions: 1920 x 1920 x 1
    POINT_CLOUD_RANGE = [-76.8, -76.8, -3.0, 76.8, 76.8, 1.0]

    MAX_POINTS_PER_VOXEL = 32
    MAX_VOXELS_TRAIN = 60000
    MAX_VOXELS_TEST = 120000

    # Calculated Grid Size (W, H, D) -> (X, Y, Z)
    GRID_SIZE = [
        int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
        int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]),
        int((POINT_CLOUD_RANGE[5] - POINT_CLOUD_RANGE[2]) / VOXEL_SIZE[2]),
    ]

    # -------------------------------------------------------------------------
    # 4. Class Definitions
    # -------------------------------------------------------------------------
    CLASS_NAMES = [
        "car",
        "truck",
        "bus",
        "bicycle",
        "pedestrian",
        "motorcycle",
        "other_vehicle",
        "emergency_vehicle",
        "animal",
    ]
    NUM_CLASSES = len(CLASS_NAMES)

    # -------------------------------------------------------------------------
    # 5. Model Architecture
    # -------------------------------------------------------------------------
    IN_CHANNELS = 4  # x, y, z, intensity

    # PointPillars Feature Net
    PFN_FILTERS = [64]

    # Backbone (ResNet-FPN)
    # Strides are relative to the input grid (1920x1920)
    # Layer 1: 1x -> 1920x1920 (Very large, usually skipped or lightweight)
    # Layer 2: 2x -> 960x960
    # Layer 3: 4x -> 480x480
    LAYER_STRIDES = [1, 2, 2]
    LAYER_FILTERS = [64, 128, 256]
    UPSAMPLE_STRIDES = [1, 2, 4]
    NUM_UPSAMPLE_FILTERS = [128, 128, 128]
    FPN_OUT_CHANNELS = 384  # Sum of upsample filters

    # Proposal Head (CenterHead) Configuration
    # Group classes into tasks to improve convergence
    TASKS = [
        dict(num_class=1, class_names=["car"]),
        dict(num_class=2, class_names=["truck", "bus"]),
        dict(num_class=2, class_names=["bicycle", "motorcycle"]),
        dict(num_class=2, class_names=["pedestrian", "animal"]),
        dict(num_class=2, class_names=["other_vehicle", "emergency_vehicle"]),
    ]

    # Head Output Channels
    # reg: (dx, dy), height: (dz), dim: (w, l, h), rot: (sin, cos), vel: (vx, vy) - no vel in this dataset
    COMMON_HEADS = {"reg": (2, 2), "height": (1, 2), "dim": (3, 2), "rot": (2, 2)}

    # Stage 2 Refinement
    ROI_HEAD_DIM = 256
    ROI_ALIGN_SIZE = 7

    # -------------------------------------------------------------------------
    # 6. Training Hyperparameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 2  # Restricted by high-resolution feature maps
    NUM_EPOCHS = 12

    # Optimizer (AdamW + OneCycle)
    LEARNING_RATE = 0.003
    WEIGHT_DECAY = 0.01
    GRAD_NORM_CLIP = 35.0

    # Loss Weights
    LOSS_CLS_WEIGHT = 1.0
    LOSS_BOX_WEIGHT = 0.25
    LOSS_IOU_WEIGHT = 1.0  # For rectification branch

    # -------------------------------------------------------------------------
    # 7. Post-Processing & Inference
    # -------------------------------------------------------------------------
    POST_CENTER_LIMIT_RANGE = [-80, -80, -5.0, 80, 80, 3.0]
    SCORE_THRESHOLD = 0.1
    NMS_IOU_THRESHOLD = 0.1  # Strict NMS for 3D boxes
    MAX_PROPOSALS = 500

    # -------------------------------------------------------------------------
    # 8. Data Augmentation
    # -------------------------------------------------------------------------
    ENABLE_GT_DATABASE = True
    DB_INFO_PATH = os.path.join(GT_DATABASE_DIR, "gt_database.parquet")
    # Minimum points required to keep a GT sample in the database
    MIN_POINTS_IN_GT = 5
