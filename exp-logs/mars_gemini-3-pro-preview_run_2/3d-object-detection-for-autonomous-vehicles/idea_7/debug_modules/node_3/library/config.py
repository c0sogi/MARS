import os
import torch
import numpy as np
import random


class Config:
    """
    Configuration for IoU-Aware DLA-CenterPoint 3D Object Detection.
    """

    # ===========================================================================
    # 1. Paths and Directories
    # ===========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working Directory for Idea 7
    WORK_DIR = "./working/idea_7"
    CACHE_DIR = os.path.join(WORK_DIR, "cache")
    LOG_DIR = os.path.join(WORK_DIR, "logs")
    CKPT_DIR = os.path.join(WORK_DIR, "checkpoints")
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    for d in [WORK_DIR, CACHE_DIR, LOG_DIR, CKPT_DIR, SUBMISSION_DIR]:
        os.makedirs(d, exist_ok=True)

    # ===========================================================================
    # 2. Reproducibility
    # ===========================================================================
    SEED = 42

    @staticmethod
    def set_seed():
        random.seed(Config.SEED)
        np.random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # ===========================================================================
    # 3. Data Configuration
    # ===========================================================================
    # Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
    # Range chosen to cover 102.4m x 102.4m area centered on ego.
    POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # Voxel Size: [x_size, y_size, z_size]
    # 0.1m resolution results in a 1024x1024 grid. Z=8.0 covers full height.
    VOXEL_SIZE = [0.1, 0.1, 8.0]

    # Multi-sweep accumulation
    NUM_SWEEPS = 3

    # Class Names (derived from EDA)
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

    # Data Loading
    NUM_WORKERS = 4

    # ===========================================================================
    # 4. Model Architecture
    # ===========================================================================
    BACKBONE_NAME = "dla34"
    PILLAR_FEATURE_DIM = 64
    DOWN_RATIO = 4  # DLA-34 typically has a stride of 4 for the first aggregation level

    # Head Configuration
    # Key: Head Name, Value: Number of Output Channels
    HEADS = {
        "hm": NUM_CLASSES,  # Heatmap for object centers
        "reg": 2,  # Local offset (x, y)
        "wh": 3,  # Dimensions (log(w), log(l), log(h))
        "rot": 2,  # Rotation (sin(yaw), cos(yaw))
        "z": 1,  # Height (z coordinate)
        "iou": 1,  # IoU Quality Prediction
    }
    HEAD_CONV = 64  # Channels for intermediate convolution in heads

    # ===========================================================================
    # 5. Training Hyperparameters
    # ===========================================================================
    BATCH_SIZE = 8
    LR = 2e-4
    WEIGHT_DECAY = 1e-2
    MAX_EPOCHS = 20

    # Loss Weights
    LOSS_WEIGHTS = {"hm": 1.0, "reg": 1.0, "wh": 0.1, "rot": 1.0, "z": 1.0, "iou": 1.0}

    # Gaussian IoU Target Generation
    MIN_IOU_TARGET = 0.0
    MAX_IOU_TARGET = 1.0

    # ===========================================================================
    # 6. Inference & Post-Processing
    # ===========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Confidence Threshold for filtering predictions
    CONF_THRESHOLD = 0.1

    # Top K predictions to keep per sample
    TOP_K = 100

    # IoU Rectification
    # Final Score = Heatmap_Score^(1 - alpha) * IoU_Score^(alpha)
    IOU_RECTIFIER_ALPHA = 0.5

    @staticmethod
    def get_grid_size():
        """
        Calculates the BEV grid size [W, H] based on range and voxel size.
        """
        x_range = Config.POINT_CLOUD_RANGE[3] - Config.POINT_CLOUD_RANGE[0]
        y_range = Config.POINT_CLOUD_RANGE[4] - Config.POINT_CLOUD_RANGE[1]
        w = int(round(x_range / Config.VOXEL_SIZE[0]))
        h = int(round(y_range / Config.VOXEL_SIZE[1]))
        return [w, h]
