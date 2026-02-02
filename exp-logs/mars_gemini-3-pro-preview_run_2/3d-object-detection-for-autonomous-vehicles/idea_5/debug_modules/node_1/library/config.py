import os
import torch
import numpy as np
import random


def set_seeds(seed=42):
    """
    Sets fixed random seeds for reproducibility across libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VoxelConfig:
    """
    Configuration for Voxelization (Pillar generation).
    """

    # Point Cloud Range: [x_min, y_min, z_min, x_max, y_max, z_max]
    # Defined in Ego-Coordinate system (meters).
    # The Z-range captures the relevant vertical extent for objects.
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    # Voxel Size: [v_x, v_y, v_z]
    # Z-voxel size matches the range height (8.0m) to create vertical pillars.
    voxel_size = [0.2, 0.2, 8.0]

    # Maximum number of points allowed per pillar
    max_points_per_pillar = 32

    # Maximum number of pillars to generate (limits memory usage)
    max_pillars_train = 16000
    max_pillars_test = 40000

    # Number of features per point in the pillar
    # Features: (x, y, z, intensity, x_c, y_c, z_c, x_p, y_p)
    num_point_features = 9

    @property
    def grid_size(self):
        """
        Calculates the grid dimensions (W, H, D) based on range and voxel size.
        Returns: [Grid_W (X), Grid_H (Y), Grid_D (Z)]
        """
        grid_w = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.voxel_size[0]
        )
        grid_h = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.voxel_size[1]
        )
        grid_d = int(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.voxel_size[2]
        )
        return [grid_w, grid_h, grid_d]


class ModelConfig:
    """
    Configuration for the Neural Network Architecture.
    """

    # Pillar Feature Net (PFN)
    num_input_features = 9
    pfn_num_filters = [64]

    # 2D Backbone (RPN / ResNet-like)
    # Input channels match the output of PFN (after scattering back to grid)
    backbone_input_channels = 64

    # Downsampling blocks configuration
    # Strides: [1, 2, 2] -> Output features at 1x, 0.5x, 0.25x resolution
    layer_strides = [1, 2, 2]
    layer_nums = [3, 5, 5]
    num_filters = [64, 128, 256]

    # Upsampling blocks configuration (FPN-like)
    # Strides: [1, 2, 4] -> Upsample all to 1x resolution
    upsample_strides = [1, 2, 4]
    num_upsample_filters = [128, 128, 128]

    # Detection Head (CenterPoint-style)
    # Classes to detect (derived from EDA)
    class_names = [
        "car",
        "truck",
        "bus",
        "pedestrian",
        "bicycle",
        "motorcycle",
        "other_vehicle",
        "emergency_vehicle",
        "animal",
    ]
    num_classes = len(class_names)

    # Head configurations: {name: num_output_channels}
    # 'hm': Heatmap (N classes)
    # 'center_z': Z-coordinate (1)
    # 'dim': Dimensions (w, l, h) (3)
    # 'rot': Rotation (sin(y), cos(y)) (2)
    # 'reg': Local center offset (x, y) (2)
    heads = {"hm": num_classes, "center_z": 1, "dim": 3, "rot": 2, "reg": 2}
    head_conv = 64  # Channels in head intermediate layers


class DataConfig:
    """
    Configuration for Data Loading, Paths, and Caching.
    """

    data_root = "./input"
    metadata_root = "./metadata"

    train_metadata_path = os.path.join(metadata_root, "train_metadata.csv")
    val_metadata_path = os.path.join(metadata_root, "val_metadata.csv")
    test_metadata_path = os.path.join(metadata_root, "test_metadata.csv")

    # Working directory for caching processed data
    work_dir = "./working/idea_5"
    cache_dir = os.path.join(work_dir, "cache")

    # Ensure directories exist
    os.makedirs(cache_dir, exist_ok=True)

    # Dataset Augmentation
    enable_augmentation = True


class TrainConfig:
    """
    Configuration for Training Loop, Optimization, and Evaluation.
    """

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training Hyperparameters
    batch_size = 4  # Adjusted for A100 memory safety
    epochs = 20
    learning_rate = 2e-3
    weight_decay = 0.01
    grad_clip_norm = 0.1

    # Learning Rate Scheduler (OneCycleLR)
    pct_start = 0.4
    div_factor = 10
    final_div_factor = 100

    # Target Generation for CenterPoint
    gaussian_overlap = 0.1
    min_radius = 2
    # Factor by which the output is downsampled relative to input grid
    # With current ModelConfig, output is 1x input resolution (512x512)
    out_size_factor = 1

    # Logging and Checkpointing
    log_interval = 50
    checkpoint_dir = os.path.join(DataConfig.work_dir, "checkpoints")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    latest_model_path = os.path.join(checkpoint_dir, "latest_model.pth")
    submission_path = "./submission/submission.csv"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)

    # Debugging / Development
    # Set to an integer (e.g., 100) to train/validate on a small subset for speed
    debug_subset_size = None


# Initialize seeds immediately
set_seeds(TrainConfig.seed)
