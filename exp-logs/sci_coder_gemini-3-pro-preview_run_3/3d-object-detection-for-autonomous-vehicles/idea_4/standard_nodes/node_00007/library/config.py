import os
import numpy as np


class Config:
    # ==============================================================================
    # PATHS & DIRECTORIES
    # ==============================================================================
    WORKING_DIR = "./working/idea_4"
    METADATA_DIR = "./metadata"
    INPUT_DIR = "./input"

    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    SUBMISSION_PATH = "./submission/submission.csv"
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "pointpillars_model.pth")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # ==============================================================================
    # DATASET CONFIGURATION
    # ==============================================================================
    # Mapping class names to integer IDs
    # Based on frequency analysis: car (83%), other_vehicle, pedestrian, bicycle, truck, bus...
    CLASS_NAMES = [
        "car",
        "truck",
        "bus",
        "other_vehicle",
        "pedestrian",
        "bicycle",
        "motorcycle",
        "emergency_vehicle",
        "animal",
    ]

    # Map class names to IDs (1-based index for model, 0 is background)
    CLASS_TO_ID = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
    ID_TO_CLASS = {i + 1: name for i, name in enumerate(CLASS_NAMES)}
    NUM_CLASSES = len(CLASS_NAMES)

    # ==============================================================================
    # VOXELIZATION & POINT CLOUD CONFIGURATION
    # ==============================================================================
    # Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
    # Covering ~100m x 100m area. Z range covers ground to high vehicles.
    POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # Voxel Size: [x_size, y_size, z_size]
    # z_size matches the full Z range to create pillars (no z-axis partition)
    VOXEL_SIZE = [0.2, 0.2, 8.0]

    # Max number of points per pillar (voxel)
    MAX_POINTS_PER_PILLAR = 32

    # Max number of pillars per sample
    MAX_PILLARS = 30000  # For training (and testing)

    # Calculated Grid Size (W, H)
    GRID_SIZE = [
        int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
        int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]),
    ]  # Should be [512, 512]

    # Input features for points: [x, y, z, intensity] usually, plus offsets added by encoder
    NUM_POINT_FEATURES = 4
    NUM_PILLAR_FEATURES = 64  # Output channels of Pillar Feature Net

    # ==============================================================================
    # ANCHOR CONFIGURATION
    # ==============================================================================
    # Anchors are defined per class or group of classes.
    # Dimensions order: [width, length, height]
    # Rotations: Radians (0 and 90 degrees)
    # Z-Center: Approximate sensor height relative to object center

    ANCHOR_GENERATOR_CONFIG = [
        {
            "class_names": ["car"],
            "anchor_sizes": [[1.93, 4.76, 1.72]],
            "anchor_rotations": [0, 1.57],
            "anchor_bottom_heights": [-1.78],
            "align_center": False,
            "feature_map_stride": 2,  # Downsampling factor of the backbone
            "matched_threshold": 0.6,
            "unmatched_threshold": 0.45,
        },
        {
            "class_names": ["truck", "bus", "other_vehicle", "emergency_vehicle"],
            "anchor_sizes": [[2.90, 10.50, 3.40]],  # Averaged large vehicle
            "anchor_rotations": [0, 1.57],
            "anchor_bottom_heights": [-1.78],
            "align_center": False,
            "feature_map_stride": 2,
            "matched_threshold": 0.55,
            "unmatched_threshold": 0.4,
        },
        {
            "class_names": ["pedestrian", "animal"],
            "anchor_sizes": [[0.77, 0.80, 1.78]],
            "anchor_rotations": [
                0
            ],  # Pedestrians don't have strong orientation, but 0 is fine
            "anchor_bottom_heights": [-1.78],
            "align_center": False,
            "feature_map_stride": 2,
            "matched_threshold": 0.5,
            "unmatched_threshold": 0.35,
        },
        {
            "class_names": ["bicycle", "motorcycle"],
            "anchor_sizes": [[0.80, 2.00, 1.50]],
            "anchor_rotations": [0, 1.57],
            "anchor_bottom_heights": [-1.78],
            "align_center": False,
            "feature_map_stride": 2,
            "matched_threshold": 0.5,
            "unmatched_threshold": 0.35,
        },
    ]

    # ==============================================================================
    # TRAINING HYPERPARAMETERS
    # ==============================================================================
    BATCH_SIZE = 4  # A100 40GB can handle this with PointPillars
    NUM_WORKERS = 4  # Number of dataloader workers
    EPOCHS = 12  # Sufficient for convergence with OneCycle
    LEARNING_RATE = 0.003
    WEIGHT_DECAY = 0.01
    GRAD_NORM_CLIP = 10.0

    # Augmentation
    AUG_ROT_RANGE = [-0.785, 0.785]  # +/- 45 degrees
    AUG_SCALE_RANGE = [0.95, 1.05]

    # ==============================================================================
    # POST-PROCESSING
    # ==============================================================================
    NMS_IOU_THRESHOLD = 0.1  # Aggressive NMS to remove duplicates
    SCORE_THRESHOLD = 0.1  # Minimum confidence to keep a box
    MAX_DETECTIONS = 100  # Max objects per sample

    # ==============================================================================
    # RANDOM SEED
    # ==============================================================================
    SEED = 42

    @staticmethod
    def set_seed():
        import torch
        import random

        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.SEED)
