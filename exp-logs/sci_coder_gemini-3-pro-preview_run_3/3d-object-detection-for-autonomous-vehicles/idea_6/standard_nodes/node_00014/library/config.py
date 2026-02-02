import os
import torch
import numpy as np
import random


class Config:
    """
    Global configuration for the Temporal PointPillars 3D Object Detection Model.
    """

    # -------------------------------------------------------------------------
    # Paths & Directories
    # -------------------------------------------------------------------------
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/idea_6"

    # Metadata paths (Pre-generated)
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Raw Data Directories (for JSON tables and sensor data)
    TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "train_data")
    TEST_DATA_DIR = os.path.join(INPUT_DIR, "test_data")
    TRAIN_LIDAR_DIR = os.path.join(INPUT_DIR, "train_lidar")
    TEST_LIDAR_DIR = os.path.join(INPUT_DIR, "test_lidar")

    # Cache & Output Directories
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    GT_DATABASE_DIR = os.path.join(WORKING_DIR, "gt_database")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model_checkpoint.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")
    LOG_FILE = os.path.join(WORKING_DIR, "train.log")

    # -------------------------------------------------------------------------
    # Hardware & Reproducibility
    # -------------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4  # Adjust based on vCPU count (12 available)
    SEED = 42

    # -------------------------------------------------------------------------
    # Dataset Parameters
    # -------------------------------------------------------------------------
    # Classes identified in data analysis
    CLASS_NAMES = [
        "car",
        "truck",
        "bus",
        "other_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "animal",
        "emergency_vehicle",
    ]
    NUM_CLASSES = len(CLASS_NAMES)
    CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

    # Point Cloud Range (Ego-centric, Meters)
    # [x_min, y_min, z_min, x_max, y_max, z_max]
    # Covering 102.4m x 102.4m area
    POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # Voxelization
    VOXEL_SIZE = [0.2, 0.2, 8.0]  # [x, y, z] - z covers full height
    MAX_POINTS_PER_PILLAR = 32
    MAX_PILLARS_TRAIN = 32000
    MAX_PILLARS_TEST = 40000

    # Input Features: x, y, z, intensity, time_lag (dt)
    NUM_POINT_FEATURES = 5

    # Temporal Aggregation
    NUM_SWEEPS = 3  # Aggregate current frame + 2 past frames

    # -------------------------------------------------------------------------
    # Model Architecture
    # -------------------------------------------------------------------------
    # PointPillars Encoder
    PILLAR_FEATURE_NET_FILTERS = [64]

    # 2D Backbone (ResNet/VGG-style FPN)
    BACKBONE_IN_CHANNELS = 64
    BACKBONE_LAYER_NUMS = [3, 5, 5]
    BACKBONE_LAYER_STRIDES = [1, 2, 2]
    BACKBONE_NUM_FILTERS = [64, 128, 256]

    # Upsampling (FPN)
    BACKBONE_UPSAMPLE_STRIDES = [1, 2, 4]
    BACKBONE_NUM_UPSAMPLE_FILTERS = [128, 128, 128]

    # Detection Head (CenterPoint)
    HEAD_CHANNELS = 64
    HEAD_TASKS = {
        "heatmap": NUM_CLASSES,  # Heatmap score
        "offset": 2,  # Center offset (x, y)
        "height": 1,  # Height (z)
        "dim": 3,  # Dimensions (l, w, h)
        "rot": 2,  # Rotation (sin, cos)
    }

    # Target Assignment
    GAUSSIAN_MIN_RADIUS = 2
    GAUSSIAN_OVERLAP = 0.1

    # -------------------------------------------------------------------------
    # Training Parameters
    # -------------------------------------------------------------------------
    BATCH_SIZE = 4  # A100 40GB allows larger batches, but start safe
    EPOCHS = 12
    LEARNING_RATE = 3e-3  # Max LR for OneCycle
    WEIGHT_DECAY = 0.01
    GRAD_CLIP_NORM = 10.0

    # Debugging
    USE_SUBSET = False  # Set True to train on a small subset
    SUBSET_SIZE = 200

    # -------------------------------------------------------------------------
    # Augmentation
    # -------------------------------------------------------------------------
    AUG_ROT_RANGE = [-0.785, 0.785]  # +/- 45 degrees
    AUG_SCALE_RANGE = [0.95, 1.05]
    AUG_TRANS_STD = [0.2, 0.2, 0.2]
    AUG_USE_GT_SAMPLING = True  # Copy-paste augmentation

    # -------------------------------------------------------------------------
    # Post-Processing
    # -------------------------------------------------------------------------
    POST_SCORE_THRESHOLD = 0.1
    POST_MAX_OBJECTS = 100
    POST_NMS_IOU = 0.1  # Low NMS for CenterPoint

    @classmethod
    def get_grid_size(cls):
        """
        Calculates the grid size (W, H, D) based on the point cloud range and voxel size.
        """
        pc_range = np.array(cls.POINT_CLOUD_RANGE)
        voxel_size = np.array(cls.VOXEL_SIZE)
        grid_size = (pc_range[3:] - pc_range[:3]) / voxel_size
        return np.round(grid_size).astype(int).tolist()

    @classmethod
    def setup(cls):
        """
        Ensures necessary directories exist and sets random seeds.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.GT_DATABASE_DIR, exist_ok=True)

        # Set seeds
        random.seed(cls.SEED)
        np.random.seed(cls.SEED)
        torch.manual_seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)

    @classmethod
    def print_config(cls):
        print("\n" + "=" * 40)
        print("MODEL CONFIGURATION")
        print("=" * 40)
        grid = cls.get_grid_size()
        print(f"{'GRID_SIZE':<30} : {grid}")
        for attr in dir(cls):
            if not attr.startswith("__") and not callable(getattr(cls, attr)):
                val = getattr(cls, attr)
                # Truncate long paths for display
                if isinstance(val, str) and len(val) > 50:
                    val = "..." + val[-47:]
                print(f"{attr:<30} : {val}")
        print("=" * 40 + "\n")
