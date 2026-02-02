import os
import torch
import random
import numpy as np


class CFG:
    """
    Configuration class for the 2.5D U-Net++ MRI Segmentation Task.
    Centralizes all hyperparameters, file paths, and model settings.
    """

    # ==============================
    # General Settings
    # ==============================
    seed = 42
    debug = False  # Set to True to run on a small subset for debugging
    debug_size = 100  # Number of samples to use when debug=True
    exp_name = "Idea5_UNetPlusPlus_EffNetB4_2.5D"
    comment = "U-Net++ with EfficientNet-B4, 2.5D input (t-1, t, t+1), Geometry-Preserving Preprocessing"

    # ==============================
    # Directory Paths
    # ==============================
    # Base Input
    base_input_dir = "./input"
    train_data_dir = os.path.join(base_input_dir, "train")
    test_data_dir = os.path.join(base_input_dir, "test")

    # Metadata (Pre-generated)
    meta_dir = "./metadata"
    train_csv = os.path.join(meta_dir, "train.csv")
    val_csv = os.path.join(meta_dir, "val.csv")
    test_csv = os.path.join(meta_dir, "test.csv")

    # Working / Output
    working_dir = "./working/idea_5"
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    predictions_dir = os.path.join(working_dir, "predictions")
    log_dir = os.path.join(working_dir, "logs")

    # Submission
    submission_dir = "./submission"
    submission_file = os.path.join(submission_dir, "submission.csv")

    # ==============================
    # Data Configuration
    # ==============================
    # Image Dimensions
    # We resize the longest dimension to 320 and pad the shorter dimension to preserve aspect ratio.
    img_size = [320, 320]

    # Input Channels
    # 2.5D Approach: Input is a stack of 3 slices [slice_idx-1, slice_idx, slice_idx+1]
    in_chans = 3

    # Classes: Large Bowel, Small Bowel, Stomach
    n_classes = 3

    # ==============================
    # Model Architecture
    # ==============================
    model_arch = "UnetPlusPlus"
    backbone = "timm-efficientnet-b4"
    encoder_weights = (
        "noisy-student"  # Pretrained weights (better than imagenet for EffNet)
    )

    # ==============================
    # Training Hyperparameters
    # ==============================
    epochs = 12
    train_batch_size = 32  # Optimized for A100 40GB with 320x320 images
    valid_batch_size = 64
    num_workers = 4  # 12 vCPUs available

    # Optimization
    lr = 2e-4
    min_lr = 1e-6
    weight_decay = 1e-2

    # Scheduler
    scheduler_type = "CosineAnnealingLR"
    T_max = epochs

    # Loss Function Weights (Composite Loss)
    # Designed to balance pixel overlap (Dice) and shape/distance (Hausdorff)
    bce_weight = 0.5
    tversky_weight = 0.5
    boundary_weight = 0.0

    # Model Options
    deep_supervision = True

    # Metric Evaluation Weights
    metric_dice_weight = 0.4
    metric_hausdorff_weight = 0.6

    # ==============================
    # Hardware & Computation
    # ==============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixed_precision = True  # Use Automatic Mixed Precision (AMP)

    # ==============================
    # Inference
    # ==============================
    mask_threshold = 0.5  # Threshold for binarizing probability maps
    min_mask_area = 0  # Minimum area filter (can be tuned during post-processing)

    @classmethod
    def setup(cls, verbose=True):
        """
        Initialize the experiment environment:
        1. Create necessary output directories.
        2. Set random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
        os.makedirs(cls.predictions_dir, exist_ok=True)
        os.makedirs(cls.log_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

        # Set seeds
        random.seed(cls.seed)
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)
        torch.cuda.manual_seed(cls.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = (
            False  # Set False for exact reproducibility, True for speed
        )

        if verbose:
            print(f"[CFG] Setup complete. Experiment: {cls.exp_name}")
            print(f"[CFG] Device: {cls.device}")
            print(f"[CFG] Output Dir: {cls.working_dir}")
