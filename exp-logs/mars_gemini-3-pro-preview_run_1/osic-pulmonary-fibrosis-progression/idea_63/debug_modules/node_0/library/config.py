import os
import random
import numpy as np
import torch


class Config:
    """
    Global configuration for the Balanced-Bottleneck Shared-Latent Network (BBSL-Net) pipeline.
    Serves as the single source of truth for hyperparameters, paths, and settings.
    """

    # ==========================================
    # 1. Reproducibility & System
    # ==========================================
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4  # Available vCPUs: 12

    # ==========================================
    # 2. Paths & Directories
    # ==========================================
    # Input Data (Read-Only)
    input_root = "./input"
    metadata_dir = "./metadata"
    train_csv = os.path.join(metadata_dir, "train.csv")
    val_csv = os.path.join(metadata_dir, "val.csv")
    test_csv = os.path.join(metadata_dir, "test.csv")

    # Working Directory (Outputs & Cache)
    working_dir = "./working/idea_63"
    cache_dir = os.path.join(working_dir, "cache")
    model_save_path = os.path.join(working_dir, "best_model.pth")

    # Submission
    submission_dir = "./submission"
    submission_path = os.path.join(submission_dir, "submission.csv")

    # ==========================================
    # 3. Data Processing & Augmentation
    # ==========================================
    image_size = 224  # Native resolution for EfficientNet-B0
    num_slices = 3  # Tri-slab configuration (Axial & Coronal)
    slab_overlap = 0.15  # 15% overlap between slabs

    # Augmentation Strategy
    # "Spatial Only": Random flips, shifts, rotations. No intensity changes.
    augmentation_mode = "spatial_only"

    # ==========================================
    # 4. Model Architecture (BBSL-Net)
    # ==========================================
    backbone = "efficientnet_b0"
    backbone_dim = 1280  # Output dimension of EfficientNet-B0 GAP (no compression)
    latent_dim = 128  # Dimension for Shared Latent Vector & Bottleneck

    # ==========================================
    # 5. Training Hyperparameters
    # ==========================================
    epochs = 30
    batch_size = 32
    learning_rate = 1e-4
    weight_decay = 1e-2

    # Optimization & Scheduling
    patience = 8  # Strict patience for Early Stopping
    scheduler_T_max = 30  # Cosine Annealing T_max

    # Metric Constraints
    max_absolute_error = 1000  # Clipped error for metric calculation
    min_confidence = 70  # Clipped confidence for metric calculation

    # ==========================================
    # 6. Debugging & Development
    # ==========================================
    debug = False  # Set to True for fast debugging runs
    debug_sample_size = 50  # Number of samples to use in debug mode

    @classmethod
    def setup(cls):
        """
        Initializes the environment:
        1. Creates necessary working directories.
        2. Sets random seeds for reproducibility.
        """
        # Create directories
        os.makedirs(cls.working_dir, exist_ok=True)
        os.makedirs(cls.cache_dir, exist_ok=True)
        os.makedirs(cls.submission_dir, exist_ok=True)

        # Set seeds
        cls._set_seed(cls.seed)

    @staticmethod
    def _set_seed(seed):
        """Sets fixed seeds for random, numpy, and torch."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# Automatically run setup when module is imported
Config.setup()
