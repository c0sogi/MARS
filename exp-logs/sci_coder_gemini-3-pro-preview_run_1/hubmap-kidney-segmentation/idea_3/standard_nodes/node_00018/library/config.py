import os
import torch
import hashlib
import json


class Config:
    # =========================================================================
    # Paths
    # =========================================================================
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"

    # Working directory for Idea 3
    WORKING_DIR = "./working/idea_3"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")

    # Metadata files
    TRAIN_METADATA_PATH = os.path.join(METADATA_DIR, "train_metadata.csv")
    VAL_METADATA_PATH = os.path.join(METADATA_DIR, "val_metadata.csv")
    TEST_METADATA_PATH = os.path.join(METADATA_DIR, "test_metadata.csv")

    # Output submission
    SUBMISSION_PATH = "./submission/submission.csv"

    # =========================================================================
    # Hardware & Reproducibility
    # =========================================================================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 12 vCPUs available
    NUM_WORKERS = 8
    SEED = 42

    # =========================================================================
    # Data Configuration
    # =========================================================================
    TILE_SIZE = 1024

    # Training Sampling Strategy
    # We sample a fixed number of tiles per epoch to define an "epoch" length
    # 8000 samples / 16 batch_size = 500 iterations per epoch
    TRAIN_NUM_SAMPLES = 8000

    # Explicit Positive Oversampling: 50% of tiles must contain glomeruli
    TRAIN_POS_RATIO = 0.5

    # Normalization (ImageNet stats)
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]

    # =========================================================================
    # Model Configuration
    # =========================================================================
    # Architecture: U-Net++ with ResNet-34
    ARCH = "UnetPlusPlus"
    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"

    IN_CHANNELS = 3
    CLASSES = 1
    ACTIVATION = None  # Output logits, apply sigmoid in loss/metric

    # =========================================================================
    # Training Configuration
    # =========================================================================
    BATCH_SIZE = 16
    EPOCHS = 15
    LR = 1e-4
    WEIGHT_DECAY = 1e-5

    # Scheduler
    T_MAX = EPOCHS  # For CosineAnnealingLR
    ETA_MIN = 1e-6

    # =========================================================================
    # Inference Configuration
    # =========================================================================
    # Threshold for binary mask
    MASK_THRESHOLD = 0.5

    # Test-Time Augmentation
    TTA_STEPS = 4  # None, HFlip, VFlip, Rotate90

    # Post-processing
    USE_ANATOMICAL_FILTER = True  # Filter prediction using Cortex ROI

    # =========================================================================
    # Utilities
    # =========================================================================
    @classmethod
    def setup(cls):
        """Creates necessary directories."""
        os.makedirs(cls.WORKING_DIR, exist_ok=True)
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(cls.SUBMISSION_PATH), exist_ok=True)

    @classmethod
    def get_params_dict(cls):
        """Returns dictionary of parameters that affect data processing/caching."""
        return {
            "tile_size": cls.TILE_SIZE,
            "train_num_samples": cls.TRAIN_NUM_SAMPLES,
            "train_pos_ratio": cls.TRAIN_POS_RATIO,
            "seed": cls.SEED,
            "norm_mean": cls.NORM_MEAN,
            "norm_std": cls.NORM_STD,
        }

    @classmethod
    def get_config_hash(cls):
        """Generates a hash based on data configuration for caching safety."""
        params = cls.get_params_dict()
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(params_str.encode("utf-8")).hexdigest()
