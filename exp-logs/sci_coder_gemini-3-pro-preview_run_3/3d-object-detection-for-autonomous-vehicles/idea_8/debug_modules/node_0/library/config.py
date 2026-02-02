import os
import torch
import numpy as np
import random


class Config:
    # ===========================================================================
    # 1. Paths & Directories
    # ===========================================================================
    DATA_ROOT = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for this specific idea/experiment
    WORK_DIR = "./working/idea_8"
    os.makedirs(WORK_DIR, exist_ok=True)

    # Sub-directories for caching
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Checkpoint path
    CHECKPOINT_PATH = os.path.join(WORK_DIR, "model_checkpoint.pth")
    SUBMISSION_PATH = os.path.join(WORK_DIR, "submission.csv")

    # Metadata Files
    TRAIN_METADATA = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA = os.path.join(METADATA_DIR, "test_metadata.csv")

    # JSON Data Paths (needed for coordinate transformations)
    # Note: These are folders containing the json files
    TRAIN_DATA_JSON = os.path.join(DATA_ROOT, "train_data")
    TEST_DATA_JSON = os.path.join(DATA_ROOT, "test_data")

    # ===========================================================================
    # 2. Dataset & Classes
    # ===========================================================================
    # Based on data analysis
    CLASSES = [
        "car",
        "truck",
        "bus",
        "trailer",
        "construction_vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
        "traffic_cone",
        "barrier",
    ]
    # Note: 'other_vehicle', 'emergency_vehicle', 'animal' appeared in analysis
    # but standard nuScenes classes are usually the ones above.
    # However, based on the provided analysis output:
    # 'car', 'other_vehicle', 'pedestrian', 'bicycle', 'truck', 'bus', 'motorcycle', 'animal', 'emergency_vehicle'
    # We will use the classes found in the dataset analysis to be safe.
    DETECTED_CLASSES = [
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
    NUM_CLASSES = len(DETECTED_CLASSES)

    CLASS_MAP = {name: i for i, name in enumerate(DETECTED_CLASSES)}

    # ===========================================================================
    # 3. Point Cloud & Voxelization (PointPillars)
    # ===========================================================================
    # Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
    # Choosing a range divisible by voxel size (0.16)
    # X: -76.8 to 76.8 (153.6m) -> 960 voxels
    # Y: -76.8 to 76.8 (153.6m) -> 960 voxels
    # Z: -5.0 to 3.0 (8m) -> Not voxelized in Z for pillars, but used for filtering
    POINT_CLOUD_RANGE = [-76.8, -76.8, -5.0, 76.8, 76.8, 3.0]

    VOXEL_SIZE = [0.16, 0.16, 8.0]  # Z voxel size = full height for pillars

    # Grid size calculated as (max-min) / size
    GRID_SIZE = [
        int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
        int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]),
        1,  # Z is 1 for pillars
    ]

    MAX_POINTS_PER_VOXEL = 32
    MAX_VOXELS_TRAIN = 16000
    MAX_VOXELS_TEST = 40000

    # Number of input features per point (x, y, z, intensity, ...)
    # Analysis showed 4 or 5. We typically use x,y,z,intensity, + geometric offsets
    NUM_POINT_FEATURES = 4

    # ===========================================================================
    # 4. Model Architecture
    # ===========================================================================
    # Backbone
    LAYER_STRIDES = [1, 2, 4]  # Downsampling steps in backbone
    NUM_FILTERS = [64, 128, 256]  # Filters in backbone blocks
    UPSAMPLE_STRIDES = [1, 2, 4]  # Deconv strides
    NUM_UPSAMPLE_FILTERS = [128, 128, 128]  # Filters after upsampling

    # Stage 1: CenterHead
    # Downsampling ratio of the backbone output relative to the input grid
    # If backbone has strides 1,2,4 and upsamples 1,2,4, the final feature map
    # usually matches the input grid resolution or is downsampled by 1 or 2.
    # Standard PointPillars often results in 2x or 4x downsampling overall.
    # Let's assume the neck concatenates upsampled features to stride 2.
    FEATURE_MAP_STRIDE = 2

    # Gaussian overlap for heatmap target generation
    GAUSSIAN_OVERLAP = 0.1
    MIN_RADIUS = 2

    # Stage 2: RoI Refinement
    ROI_SIZE = 7  # 7x7 pooling
    ROI_OUT_CHANNELS = 256

    # Code weights for box regression [x, y, z, w, l, h, sin, cos]
    CODE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # ===========================================================================
    # 5. Training Hyperparameters
    # ===========================================================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 4
    NUM_WORKERS = 4

    # Optimization
    LEARNING_RATE = 0.003
    WEIGHT_DECAY = 0.01
    GRAD_NORM_CLIP = 10.0

    # Scheduling
    NUM_EPOCHS = 15  # Adjusted for 24h limit
    PCT_START = 0.4  # OneCycleLR parameter
    DIV_FACTOR = 10
    FINAL_DIV_FACTOR = 100

    # Loss Weights
    LOSS_WEIGHT_HM = 1.0
    LOSS_WEIGHT_BOX = 0.25
    LOSS_WEIGHT_REFINE = 1.0

    # ===========================================================================
    # 6. Inference & Post-Processing
    # ===========================================================================
    SCORE_THRESHOLD = 0.1
    POST_CENTER_LIMIT_RANGE = [-80, -80, -10, 80, 80, 10]
    NMS_IOU_THRESHOLD = 0.1  # Stricter NMS for 3D
    PRE_MAX_SIZE = 1000
    POST_MAX_SIZE = 100

    # ===========================================================================
    # 7. Reproducibility
    # ===========================================================================
    SEED = 42

    @staticmethod
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Set seed immediately upon import
Config.set_seed(Config.SEED)
