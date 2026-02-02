import os
import torch
import numpy as np


class Config:
    """
    Central configuration for the 3D Object Detection Pipeline.
    """

    # ==============================================================================
    # 1. Global & Reproducibility
    # ==============================================================================
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # Number of dataloader workers

    # ==============================================================================
    # 2. File Paths & Directories
    # ==============================================================================
    # Input Directories (Read-Only)
    INPUT_DIR = "./input"
    TRAIN_DATA_DIR = os.path.join(INPUT_DIR, "train_data")
    TEST_DATA_DIR = os.path.join(INPUT_DIR, "test_data")

    # Metadata Directories (Pre-generated)
    METADATA_DIR = "./metadata"
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Working Directories (Write Allowed)
    # Specific directory for Idea 6 caching as requested
    WORKING_DIR = "./working/idea_6"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
    LOG_DIR = os.path.join(WORKING_DIR, "logs")

    # Submission
    SUBMISSION_DIR = "./submission"
    SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, "submission.csv")

    # ==============================================================================
    # 3. Data & Voxelization Parameters
    # ==============================================================================
    # Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
    # Covers a 102.4m x 102.4m area centered roughly on the ego vehicle
    POINT_CLOUD_RANGE = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # Voxel Size: [x_size, y_size, z_size]
    # z_size = 8.0 matches the full z-range (3.0 - (-5.0)), creating Pillars
    VOXEL_SIZE = [0.2, 0.2, 8.0]

    # Maximum number of points per pillar (for Scatter/Pillar Feature Net)
    MAX_POINTS_PER_PILLAR = 32

    # Maximum number of pillars to keep (limits memory usage)
    MAX_PILLARS = 30000

    # Number of features per point input to the network
    # (x, y, z, intensity, time_lag) -> 5 features
    NUM_POINT_FEATURES = 5

    # Temporal Context
    MAX_SWEEPS = 3  # Current frame + 2 previous frames

    # ==============================================================================
    # 4. Model Hyperparameters
    # ==============================================================================
    # Class Names (Order matters for prediction mapping)
    CLASS_NAMES = [
        "car",
        "truck",
        "bus",
        "emergency_vehicle",
        "other_vehicle",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "animal",
    ]
    NUM_CLASSES = len(CLASS_NAMES)

    # Pillar Feature Net
    PFN_OUT_CHANNELS = 64  # Output channels after Pillar encoding

    # Backbone (U-Net / ResNet)
    BACKBONE_IN_CHANNELS = 64
    BACKBONE_LAYER_STRIDES = [1, 2, 2]
    BACKBONE_LAYER_CHANNELS = [64, 128, 256]

    # Detection Head
    HEAD_HIDDEN_CHANNELS = 64

    # Grid Size Calculation (Derived)
    # X: 102.4 / 0.2 = 512
    # Y: 102.4 / 0.2 = 512
    GRID_SIZE = [
        int((POINT_CLOUD_RANGE[3] - POINT_CLOUD_RANGE[0]) / VOXEL_SIZE[0]),
        int((POINT_CLOUD_RANGE[4] - POINT_CLOUD_RANGE[1]) / VOXEL_SIZE[1]),
        1,  # Z dimension is 1 for Pillars
    ]

    # ==============================================================================
    # 5. Training Parameters
    # ==============================================================================
    BATCH_SIZE = 4  # Adjusted for A100 memory with temporal sweeps
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.003
    WEIGHT_DECAY = 0.01
    GRAD_CLIP_NORM = 10.0

    # Debug / Development options
    DEBUG = False
    DEBUG_SAMPLE_SIZE = 100  # Number of samples to use in debug mode

    # ==============================================================================
    # 6. Post-Processing & Evaluation
    # ==============================================================================
    CONFIDENCE_THRESHOLD = 0.1
    NMS_IOU_THRESHOLD = 0.2
    MAX_DETECTIONS = 500

    # Evaluation Metric Config
    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    @classmethod
    def setup(cls):
        """
        Creates necessary working directories.
        """
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.SUBMISSION_DIR, exist_ok=True)

        # Set deterministic behavior
        torch.manual_seed(cls.SEED)
        np.random.seed(cls.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cls.SEED)

    @classmethod
    def get_dataset_config(cls, is_train=True):
        """
        Returns a dictionary configuration for the dataset loader.
        """
        return {
            "data_dir": cls.INPUT_DIR,
            "metadata_path": (
                cls.TRAIN_METADATA_PATH if is_train else cls.VAL_METADATA_PATH
            ),
            "max_sweeps": cls.MAX_SWEEPS,
            "point_cloud_range": cls.POINT_CLOUD_RANGE,
            "voxel_size": cls.VOXEL_SIZE,
            "max_points_per_pillar": cls.MAX_POINTS_PER_PILLAR,
            "max_pillars": cls.MAX_PILLARS,
            "num_features": cls.NUM_POINT_FEATURES,
            "class_names": cls.CLASS_NAMES,
            "augment": is_train,  # Enable augmentation only for training
            "debug_limit": cls.DEBUG_SAMPLE_SIZE if cls.DEBUG else None,
        }
