import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration module for the Rasterized BEV-YOLO 3D Object Detection pipeline.
    """

    # ==============================================================================
    # 1. Paths & Directories
    # ==============================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for artifacts (checkpoints, logs, cache)
    WORKING_DIR = "./working/idea_3"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Cache directory for processed data (e.g. rasterized BEV images)
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Checkpoint directory
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Metadata File Paths
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Submission Output
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================================================================
    # 2. Data Parameters (BEV Rasterization)
    # ==============================================================================
    # Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
    # We define a square Region of Interest (ROI) centered at the ego vehicle.
    # A range of +/- 51.2m with 0.2m resolution results in a 512x512 grid.
    X_MIN, Y_MIN, Z_MIN = -51.2, -51.2, -5.0
    X_MAX, Y_MAX, Z_MAX = 51.2, 51.2, 3.0
    PC_RANGE = [X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX]

    # Voxel (Grid) Size in meters [x_res, y_res]
    VOXEL_SIZE = [0.2, 0.2]

    # Calculated Grid Dimensions (Height, Width)
    BEV_HEIGHT = int((Y_MAX - Y_MIN) / VOXEL_SIZE[1])  # 512
    BEV_WIDTH = int((X_MAX - X_MIN) / VOXEL_SIZE[0])  # 512

    # Input Feature Channels for the Network:
    # 1. Max Height (geometry)
    # 2. Mean Intensity (reflectance)
    # 3. Point Density (occupancy)
    IN_CHANNELS = 3

    # ==============================================================================
    # 3. Model Architecture
    # ==============================================================================
    BACKBONE = "resnet18"

    # Target Classes (derived from EDA)
    DETECT_CLASSES = [
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
    NUM_CLASSES = len(DETECT_CLASSES)
    CLASS_MAP = {name: i for i, name in enumerate(DETECT_CLASSES)}

    # Anchor Boxes for the Detection Head (Width, Length) in meters.
    # These priors are based on the mean dimensions of objects found in the EDA.
    ANCHORS = [
        [1.9, 4.9],  # Car / Other Vehicle
        [0.6, 0.8],  # Pedestrian
        [2.9, 10.0],  # Truck / Bus
        [0.8, 2.0],  # Bicycle / Motorcycle
        [0.5, 0.5],  # Small object / Animal
    ]
    NUM_ANCHORS = len(ANCHORS)

    # ==============================================================================
    # 4. Training Hyperparameters
    # ==============================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loader Settings
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Optimizer Settings
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    NUM_EPOCHS = 15

    # Scheduler Settings
    WARMUP_EPOCHS = 1

    # Early Stopping
    PATIENCE = 3

    # Inference Thresholds
    CONF_THRESHOLD = 0.1  # Minimum confidence to keep a prediction
    NMS_IOU_THRESHOLD = 0.2  # IoU threshold for Non-Maximum Suppression

    # ==============================================================================
    # 5. Global Setup & Utilities
    # ==============================================================================
    @staticmethod
    def set_seed(seed=42):
        """
        Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Automatically set seed upon import
Config.set_seed(Config.SEED)
